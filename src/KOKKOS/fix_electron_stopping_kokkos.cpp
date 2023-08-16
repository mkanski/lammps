/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Electronic stopping power
   Contributing authors: K. Avchaciov and T. Metspalu
   Information: k.avchachov@gmail.com
------------------------------------------------------------------------- */

#include "fix_electron_stopping_kokkos.h"

#include "atom_kokkos.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "region.h"
#include "update.h"

#include "atom_masks.h"
#include "kokkos_base.h"
#include "neigh_list_kokkos.h"

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixElectronStoppingKokkos<DeviceType>::FixElectronStoppingKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixElectronStopping(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixElectronStoppingKokkos<DeviceType>::~FixElectronStoppingKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixElectronStoppingKokkos<DeviceType>::init()
{
  //FixElectronStopping::init();
  SeLoss = 0.0;
  SeLoss_sync_flag = 0;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                         !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  request->enable_full();

  if (utils::strmatch(update->integrate_style,"^respa"))
    error->all(FLERR,"Cannot (yet) use respa with Kokkos");

  k_elstop_ranges = Kokkos::DualView<double**, Kokkos::LayoutRight, DeviceType>("FixElectronStopping:elstop_ranges", atom->ntypes+1, table_entries);
  for (int i = 0; i <= atom->ntypes; i++)
    for (int j = 0; j < table_entries; j++) k_elstop_ranges.h_view(i,j) = elstop_ranges[i][j];
  k_elstop_ranges.template modify<LMPHostType>();
  k_elstop_ranges.template sync<DeviceType>();

  k_table_entries = Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType>("FixElectronStopping:table_entries");
  k_table_entries.h_view() = table_entries;
  k_table_entries.template modify<LMPHostType>();
  k_table_entries.template sync<DeviceType>();

  k_mvv2e = Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType>("FixElectronStopping:mvv2e");
  k_mvv2e.h_view() = force->mvv2e;
  k_mvv2e.template modify<LMPHostType>();
  k_mvv2e.template sync<DeviceType>();
  
  k_minneigh = Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType>("FixElectronStopping:minneigh");
  k_minneigh.h_view() = minneigh;
  k_minneigh.template modify<LMPHostType>();
  k_minneigh.template sync<DeviceType>();

  k_Ecut = Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType>("FixElectronStopping:Ecut");
  k_Ecut.h_view() = Ecut;
  k_Ecut.template modify<LMPHostType>();
  k_Ecut.template sync<DeviceType>();

}

template<class DeviceType>
void FixElectronStoppingKokkos<DeviceType>::post_force(int)
{
  SeLoss_sync_flag = 0;
  
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();

  if (rmass.data())
    atomKK->sync(execution_space, V_MASK | F_MASK | MASK_MASK | RMASS_MASK);
  else
    atomKK->sync(execution_space, V_MASK | F_MASK | MASK_MASK | TYPE_MASK);
  
  int inum = list->inum;

  double curr_SeLoss;

  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixElectronStopping>(0, inum), *this, curr_SeLoss);
  copymode = 0;

  SeLoss += curr_SeLoss*update->dt;

  atomKK->modified(execution_space, F_MASK);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixElectronStoppingKokkos<DeviceType>::operator()(TagFixElectronStopping, const int &ii, double& curr_SeLoss) const {

  int i = d_ilist[ii];
  // Do fast checks first, only then the region check
  if (!(mask[i] & groupbit))
    return;
  

  // Avoid atoms outside bulk material
  if (d_numneigh(i) < k_minneigh.d_view())
    return;

  int itype = type[i];
  double massone = (rmass.data()) ? rmass(i) : mass(itype);
  double v2 = v(i,0) * v(i,0) + v(i,1) * v(i,1) + v(i,2) * v(i,2);
  double energy = 0.5 * k_mvv2e.d_view() * massone * v2;

  if (energy < k_Ecut.d_view()) return;
  if (energy < k_elstop_ranges.d_view(0,0)) return;
//  if (energy > k_elstop_ranges.d_view(0,k_table_entries.d_view()-1))
//    error->one(FLERR, "Fix electron/stopping: kinetic energy too high for atom {}: {} vs {}",
//                 tag[i], energy, k_elstop_ranges.d_view(0,table_entries - 1));
  
  // Binary search to find correct energy range
  int iup = k_table_entries.d_view() - 1;
  int idown = 0;
  
  while (true) {
    int ihalf = idown + (iup - idown) / 2;
    if (ihalf == idown) break;
    if (k_elstop_ranges.d_view(0,ihalf) < energy)
      idown = ihalf;
    else
      iup = ihalf;
  }
  
  double Se_lo = k_elstop_ranges.d_view(itype,idown);
  double Se_hi = k_elstop_ranges.d_view(itype,iup);
  double E_lo = k_elstop_ranges.d_view(0,idown);
  double E_hi = k_elstop_ranges.d_view(0,iup);

  // Get electronic stopping with a simple linear interpolation
  double Se = (Se_hi - Se_lo) / (E_hi - E_lo) * (energy - E_lo) + Se_lo;

  double vabs = sqrt(v2);
  double factor = -Se / vabs;

  f(i,0) += v(i,0) * factor;
  f(i,1) += v(i,1) * factor;
  f(i,2) += v(i,2) * factor;

  curr_SeLoss = Se * vabs; //multiplication by dt after reduce
   
}

namespace LAMMPS_NS {
template class FixElectronStoppingKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixElectronStoppingKokkos<LMPHostType>;
#endif
}

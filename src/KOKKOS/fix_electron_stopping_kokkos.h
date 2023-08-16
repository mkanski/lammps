/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(electron/stopping/kk,FixElectronStoppingKokkos<LMPDeviceType>);
FixStyle(electron/stopping/kk/device,FixElectronStoppingKokkos<LMPDeviceType>);
FixStyle(electron/stopping/kk/host,FixElectronStoppingKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_ELECTRON_STOPPING_KOKKOS_H
#define LMP_FIX_ELECTRON_STOPPING_KOKKOS_H

#include "fix_electron_stopping.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixElectronStopping{};

template<class DeviceType>
class FixElectronStoppingKokkos : public FixElectronStopping {
 public:
  typedef ArrayTypes<DeviceType> AT;

  FixElectronStoppingKokkos(class LAMMPS *, int, char **);
  ~FixElectronStoppingKokkos() override;
  void init() override;
//  int setmask() override;
  void post_force(int) override;
//  void init_list(int, class NeighList *) override;
//  double compute_scalar() override;

   KOKKOS_INLINE_FUNCTION
   void operator()(TagFixElectronStopping, const int&, double&) const;

 private:
  typename AT::t_v_array v;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_int_1d_randomread type;
  typename AT::t_float_1d mass;
  typename AT::t_float_1d rmass;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  Kokkos::DualView<double**, Kokkos::LayoutRight, DeviceType> k_elstop_ranges;
  Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType> k_table_entries;
  Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType> k_mvv2e;
  Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType> k_minneigh;
  Kokkos::DualView<int, Kokkos::LayoutRight, DeviceType> k_Ecut;

/*
  void read_table(const char *);
  void grow_table();

  double Ecut;                  // cutoff energy
  double SeLoss, SeLoss_all;    // electronic energy loss
  int SeLoss_sync_flag;         // sync done since last change?

  int maxlines;              // max number of lines in table
  int table_entries;         // number of table entries actually read
  double **elstop_ranges;    // [ 0][i]: energies
                             // [>0][i]: stopping powers per type

  char *idregion;          // region id
  class Region *region;    // region pointer if used, else NULL
  int minneigh;            // minimum number of neighbors

  class NeighList *list;
*/
};

}    // namespace LAMMPS_NS

#endif
#endif

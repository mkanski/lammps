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
#include "neighbor.h"
#include "potential_file_reader.h"
#include "region.h"
#include "update.h"

#include "kokkos_base.h"

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixElectronStoppingKokkos<DeviceType>::FixElectronStoppingKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixElectronStopping(lmp, narg, arg) //, elstop_ranges(nullptr), idregion(nullptr), region(nullptr), list(nullptr)
{
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixElectronStoppingKokkos<DeviceType>::~FixElectronStoppingKokkos()
{
}

namespace LAMMPS_NS {
template class FixElectronStoppingKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixElectronStoppingKokkos<LMPHostType>;
#endif
}

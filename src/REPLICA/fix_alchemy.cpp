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

#include "fix_alchemy.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "universe.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

static double get_lambda(const bigint &step, const bigint &begin, const bigint &end, int iworld)
{
  double lambda = step - begin;
  if (lambda != 0.0) lambda /= end - begin;
  if (iworld == 0) lambda = 1.0 - lambda;
  return lambda;
}

/* ---------------------------------------------------------------------- */

FixAlchemy::FixAlchemy(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), commbuf(nullptr)
{
  if (narg != 3) error->all(FLERR, "Incorrect number of arguments for fix alchemy");

  lambda = epot[0] = epot[1] = 0.0;
  for (int i = 0; i < 6; ++i) pressure[i] = 0.0;

  no_change_box = 1;
  vector_flag = 1;
  size_vector = 10;
  extvector = 0;
  ilevel_respa = 0;
  nmax = 6;

  // set up rank-to-rank communicator
  if (universe->nworlds != 2) error->all(FLERR, "Must use exactly two partitions");
  int color = comm->me;
  int key = universe->iworld;
  MPI_Comm_split(universe->uworld, color, key, &samerank);

  // check that we have the same domain decomposition on all ranks
  int my_nlocal[2] = {0, 0};
  int all_nlocal[2] = {0, 0};
  my_nlocal[universe->iworld] = atom->nlocal;
  MPI_Allreduce(my_nlocal, all_nlocal, 2, MPI_INT, MPI_SUM, samerank);
  int fail = (all_nlocal[0] == all_nlocal[1]) ? 0 : 1;
  int allfail = 0;
  MPI_Allreduce(&fail, &allfail, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (allfail)
    error->all(FLERR, "Number of atoms and domain decomposition must match for both partitions");

  id_pe = std::string(id) + "_pe";
  pe = modify->add_compute(id_pe + " all pe");
  pe->addstep(update->ntimestep);
  id_temp = std::string(id) + "_temp";
  temp = modify->add_compute(id_temp + " all temp");
  temp->addstep(update->ntimestep);
  id_press = std::string(id) + "_press";
  press = modify->add_compute(id_press + " all pressure " + id_temp);
  press->addstep(update->ntimestep);
}

/* ---------------------------------------------------------------------- */

FixAlchemy::~FixAlchemy()
{
  MPI_Comm_free(&samerank);
  modify->delete_compute(id_pe);
  modify->delete_compute(id_temp);
  modify->delete_compute(id_press);
  memory->destroy(commbuf);
}

/* ---------------------------------------------------------------------- */

int FixAlchemy::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAlchemy::init()
{
  int onenmax = MAX(nmax, 3 * atom->nmax);
  MPI_Allreduce(&onenmax, &nmax, 1, MPI_INT, MPI_MAX, universe->uworld);
  memory->destroy(commbuf);
  memory->create(commbuf, sizeof(double) * nmax, "alchemy:nmax");
}

/* ---------------------------------------------------------------------- */

void FixAlchemy::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    respa->copy_f_flevel(ilevel_respa);
  } else {
    post_force(vflag);
  }
}

/* ---------------------------------------------------------------------- */

void FixAlchemy::post_integrate()
{
  const int nall = atom->nlocal + atom->nghost;
  MPI_Bcast(&atom->x[0][0], 3 * nall, MPI_DOUBLE, 0, samerank);
}

/* ---------------------------------------------------------------------- */

void FixAlchemy::post_force(int /*vflag*/)
{
  if (3 * atom->nmax > nmax) {
    nmax = 3 * atom->nmax;
    memory->grow(commbuf, sizeof(double) * atom->nmax, "alchemy:commbuf");
  }

  const int nall = 3 * atom->nlocal;
  double *f = &atom->f[0][0];
  lambda = get_lambda(update->ntimestep, update->beginstep, update->endstep, universe->iworld);

  for (int i = 0; i < nall; ++i) commbuf[i] = f[i] * lambda;
  MPI_Allreduce(commbuf, f, nall, MPI_DOUBLE, MPI_SUM, samerank);

  // sum up potential energy
  const double scalefac = 1.0 / comm->nprocs;
  commbuf[0] = commbuf[1] = commbuf[2] = 0.0;
  commbuf[universe->iworld] = scalefac * pe->compute_scalar();
  commbuf[2] = lambda * scalefac * pe->compute_scalar();
  MPI_Allreduce(commbuf, epot, 3, MPI_DOUBLE, MPI_SUM, universe->uworld);
  pe->addstep(update->ntimestep + 1);

  // sum up pressure
  press->compute_vector();
  for (int i = 0; i < 6; ++i) commbuf[i] = lambda * scalefac * press->vector[i];
  MPI_Allreduce(commbuf, pressure, 6, MPI_DOUBLE, MPI_SUM, universe->uworld);
  press->addstep(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

double FixAlchemy::compute_vector(int n)
{
  if (n == 0) return lambda;
  if ((n > 0) && (n < 4)) return epot[n - 1];
  return pressure[n - 4];
}

/* ---------------------------------------------------------------------- */

void *FixAlchemy::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "lambda") == 0) { return &lambda; }
  if (strcmp(str, "pe") == 0) { return &epot[2]; }
  dim = 1;
  if (strcmp(str, "pressure") == 0) { return pressure; }
  return nullptr;
}

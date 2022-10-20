/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(ttm/grid,FixTTMGrid);
// clang-format on
#else

#ifndef LMP_FIX_TTM_GRID_H
#define LMP_FIX_TTM_GRID_H

#include "fix_ttm.h"

namespace LAMMPS_NS {

class FixTTMGrid : public FixTTM {
 public:
  FixTTMGrid(class LAMMPS *, int, char **);
  ~FixTTMGrid() override;
  void post_constructor() override;
  void init() override;
  void post_force(int) override;
  void end_of_step() override;
  void write_restart_file(const char *) override;
  double compute_vector(int) override;
  double memory_usage() override;

  // grid communication

  void reset_grid() override;

  void pack_forward_grid(int, void *, int, int *) override;
  void unpack_forward_grid(int, void *, int, int *) override;
  void pack_reverse_grid(int, void *, int, int *) override;
  void unpack_reverse_grid(int, void *, int, int *) override;
  void pack_remap_grid(void *, int, int *) override;
  void unpack_remap_grid(void *, int, int *) override;
  int unpack_read_grid(char *) override;
  void pack_write_grid(int, void *) override;
  void unpack_write_grid(int, void *, int *) override;

  int get_grid_by_name(const std::string &, int &) override;
  void *get_grid_by_index(int) override;
  int get_griddata_by_name(int, const std::string &, int &) override;
  void *get_griddata_by_index(int) override;

 private:
  int ngridown, ngridout;
  int nxlo_in, nxhi_in, nylo_in, nyhi_in, nzlo_in, nzhi_in;
  int nxlo_out, nxhi_out, nylo_out, nyhi_out, nzlo_out, nzhi_out;
  double delxinv, delyinv, delzinv;
  double skin_original;
  FILE *fpout;

  class Grid3d *grid;
  class Grid3d *grid_previous;
  double ***T_electron_previous;
  int ngrid_buf1, ngrid_buf2;
  double *grid_buf1, *grid_buf2;

  double ***T_electron_read;

  void allocate_grid() override;
  void deallocate_grid() override;
  void read_electron_temperatures(const std::string &) override;
};

}    // namespace LAMMPS_NS

#endif
#endif

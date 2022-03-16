/* -*- c++ -*- -----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------ */

#ifndef LMP_LATBOLTZ_CONST_H
#define LMP_LATBOLTZ_CONST_H

#include "math_const.h"
using LAMMPS_NS::MathConst::MY_ISQRT2;
using LAMMPS_NS::MathConst::MY_SQRT2;

static constexpr double kappa_lb = 0.0;

// 15-velocity lattice propogation vectors
static constexpr int e15[15][3] = {{0, 0, 0},  {1, 0, 0},  {0, 1, 0},   {-1, 0, 0},   {0, -1, 0},
                                   {0, 0, 1},  {0, 0, -1}, {1, 1, 1},   {-1, 1, 1},   {-1, -1, 1},
                                   {1, -1, 1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, -1}, {1, -1, -1}};

// 15-velocity weights
static constexpr double w_lb15[15] = {2. / 9.,  1. / 9.,  1. / 9.,  1. / 9.,  1. / 9.,
                                      1. / 9.,  1. / 9.,  1. / 72., 1. / 72., 1. / 72.,
                                      1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72.};

// 15-velocity normalizations
static constexpr double Ng_lb15[15] = {1., 3., 3.,       3.,       9. / 2.,  9. / 2., 9. / 2., 9.,
                                       9., 9., 27. / 2., 27. / 2., 27. / 2., 9.,      1.};

// 15-velcity transformation matrix for f_i to moments
// clang-format off
static constexpr double mg_lb15[15][15] = {
  {     1.,        1.,        1.,        1.,        1.,        1.,        1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.},
  {     0.,        1.,        0.,       -1.,        0.,        0.,        0.,     1.,    -1.,    -1.,     1.,     1.,    -1.,    -1.,     1.},
  {     0.,        0.,        1.,        0.,       -1.,        0.,        0.,     1.,     1.,    -1.,    -1.,     1.,     1.,    -1.,    -1.},
  {     0.,        0.,        0.,        0.,        0.,        1.,       -1.,     1.,     1.,     1.,     1.,    -1.,    -1.,    -1.,    -1.},
  { -1./3.,     2./3.,    -1./3.,     2./3.,    -1./3.,    -1./3.,    -1./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.},
  { -1./3.,    -1./3.,     2./3.,    -1./3.,     2./3.,    -1./3.,    -1./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.},
  { -1./3.,    -1./3.,    -1./3.,    -1./3.,    -1./3.,     2./3.,     2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.,  2./3.},
  {     0.,        0.,        0.,        0.,        0.,        0.,        0.,     1.,    -1.,     1.,    -1.,     1.,    -1.,     1.,    -1.},
  {     0.,        0.,        0.,        0.,        0.,        0.,        0.,     1.,     1.,    -1.,    -1.,    -1.,    -1.,     1.,     1.},
  {     0.,        0.,        0.,        0.,        0.,        0.,        0.,     1.,    -1.,    -1.,     1.,    -1.,     1.,     1.,    -1.},
  {     0.,        0.,    -1./3.,        0.,     1./3.,        0.,        0.,  2./3.,  2./3., -2./3., -2./3.,  2./3.,  2./3., -2./3., -2./3.},
  {     0.,        0.,        0.,        0.,        0.,    -1./3.,     1./3.,  2./3.,  2./3.,  2./3.,  2./3., -2./3., -2./3., -2./3., -2./3.},
  {     0.,    -1./3.,        0.,     1./3.,        0.,        0.,        0.,  2./3., -2./3., -2./3.,  2./3.,  2./3., -2./3., -2./3.,  2./3.},
  {     0.,        0.,        0.,        0.,        0.,        0.,        0.,     1.,    -1.,     1.,    -1.,    -1.,     1.,    -1.,     1.},
  {MY_SQRT2,-MY_ISQRT2,-MY_ISQRT2,-MY_ISQRT2,-MY_ISQRT2,-MY_ISQRT2,-MY_ISQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2,MY_SQRT2}
};
// clang-format on

// 15-velocity opposite lattice directions for bounce-back, i.e. od[i] = j such that e15[j]=-e15[i]
static constexpr int od[15] = {0, 3, 4, 1, 2, 6, 5, 13, 14, 11, 12, 9, 10, 7, 8};

// 15-velocity bounce-back list
// bbl[i][0] = number of bounce-back directions for orientation i
// bbl[i][j]...bbl[i][bbl[i][0]] directions that would be coming from inside the wall so need to come from bounce-back
// bbl[i][[bbl[i][0]+1]...bbl[i][16] directions where standard propogation can proceed (pointing into or along wall)
// inside edge has 1/4 inside domain, 3/4 outside domain
// outside edge has 3/4 outside domain, 1/4 inside domain
// Note: 1. This list is not exhaustive (eg. there should be 12 inside and 12 outside edges possible, it just covers cases
//          accessible in the pit routines.  Could be generalized to include other geometries
//       2. Need better labelling for corners (particularly in-out) that distinguishes different cases (e.g. 10 and 29 are NOT same, also 11,31)
// ori   wall normals (point into domain)
//  0    not relevent, ori==0 only for lattice type 0 (standard bulk fluid) and 2 (outside domain)
//  1    wall +x
//  2    wall +y
//  3    wall +z
//  4    outside edge +x,+z
//  5    inside edge  +x,+z
//  6    inside edge +y,+z
//  7    outside edge  +y,+z
//  8    inside edge -x,-y
//  9    inside edge -x,+y
// 10    in-out corner +x,+y,+z
// 11    in-out corner +x,-y,+z
// 12    inside corner -x,+y,+z
// 13    inside corner -x,-y,+z
// 14    wall -x
// 15    wall -y
// 16    wall -z
// 17    outside edge -x,+z
// 18    inside edge  -x,+z
// 19    inside edge -y,+z
// 20    outside edge  -y,+z
// 21    inside edge +x,-y
// 22    inside edge +x,+y
// 23    in-out corner -x,+y,+z
// 24    in-out corner -x,-y,+z
// 25    inside corner +x,+y,+z
// 26    inside corner +x,-y,+z
// 27    inside edge +y,-z
// 28    inside edge -y,-z
// 29    in-out corner +x,+y,+z
// 30    in-out corner +x,-y,+z
// 31    in-out corner -x,+y,+z
// 32    in-out corner -x,-y,+z
// clang-format off
static constexpr int bbl[33][16] = {
  { 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
  { 5,  1,  7, 10, 11, 14,      0,  2,  3,  4,  5,  6,  8,  9, 12, 13},
  { 5,  2,  7,  8, 11, 12,      0,  1,  3,  4,  5,  6,  9, 10, 13, 14},
  { 5,  5,  7,  8,  9, 10,      0,  1,  2,  3,  4,  6, 11, 12, 13, 14},
  { 4,  7, 10,  1,  5,      0,  2,  3,  4,  6,  8,  9, 11, 12, 13, 14},
  { 8,  1,  5,  7, 10,  8,  9, 11, 14,      0,  2,  3,  4,  6, 12, 13},
  { 8,  2,  5,  7,  8,  9, 10, 11, 12,      0,  1,  3,  4,  6, 13, 14},
  { 4,  2,  5,  7,  8,      0,  1,  3,  4,  6,  9, 10, 11, 12, 13, 14},
  { 8,  3,  4,  9, 13,  8, 10, 12, 14,      0,  1,  2,  5,  6,  7, 11},
  { 8,  2,  3,  8, 12,  7,  9, 11, 13,      0,  1,  4,  5,  6, 10, 14},
  { 3,  7,  8, 10,      0,  1,  2,  3,  4,  5,  6,  9, 11, 12, 13, 14},
  { 3,  7,  9, 10,      0,  1,  2,  3,  4,  5,  6,  8, 11, 12, 13, 14},
  {10,  2,  3,  5,  8,  7,  9, 10, 11, 12, 13,      0,  1,  4,  6, 14},
  {10,  3,  4,  5,  9,  7,  8, 10, 12, 13, 14,      0,  1,  2,  6, 11},
  { 5,  3,  8,  9, 12, 13,      0,  1,  2,  4,  5,  6,  7, 10, 11, 14},
  { 5,  4,  9, 10, 13, 14,      0,  1,  2,  3,  5,  6,  7,  8, 11, 12},
  { 5,  6, 11, 12, 13, 14,      0,  1,  2,  3,  4,  5,  7,  8,  9, 10},
  { 4,  8,  9,  3,  5,      0,  1,  2,  4,  6,  7, 10, 11, 12, 13, 14},
  { 8,  3,  5,  8,  9,  7, 10, 12, 13,      0,  1,  2,  4,  6, 11, 14},
  { 8,  4,  5,  9, 10,  7,  8, 13, 14,      0,  1,  2,  3,  6, 11, 12},
  { 4,  4,  5,  9, 10,      0,  1,  2,  3,  6,  7,  8, 11, 12, 13, 14},
  { 8,  1,  4, 10, 14,  7,  9, 11, 13,      0,  2,  3,  5,  6,  8, 12},
  { 8,  1,  2,  7, 11,  8, 10, 12, 14,      0,  3,  4,  5,  6,  9, 13},
  { 3,  7,  8,  9,      0,  1,  2,  3,  4,  5,  6, 10, 11, 12, 13, 14},
  { 3,  8,  9, 10,      0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14},
  {10,  1,  2,  5,  7,  8,  9, 10, 11, 12, 14,      0,  3,  4,  6, 13},
  {10,  1,  4,  5, 10,  7,  8,  9, 11, 13, 14,      0,  2,  3,  6, 12},
  { 8,  2,  6, 11, 12,  7,  8, 13, 14,      0,  1,  3,  4,  5,  9, 10},
  { 8,  4,  6, 13, 14,  9, 10, 11, 12,      0,  1,  2,  3,  5,  7,  8},
  { 6,  2,  7,  8, 11, 10, 12,      0,  1,  3,  4,  5,  6,  9, 13, 14},
  { 6,  4,  9, 10, 14,  7, 13,      0,  1,  2,  3,  5,  6,  8, 11, 12},
  { 6,  2,  7,  8, 12,  9, 11,      0,  1,  3,  4,  5,  6, 10, 13, 14},
  { 6,  4,  9, 10, 13,  8, 14,      0,  1,  2,  3,  5,  6,  7, 11, 12}
};
//clang-format on

// 19-velocity lattice propogation vectors
static constexpr int e19[19][3] = {{0, 0, 0},   {1, 0, 0},  {0, 1, 0},  {-1, 0, 0}, {0, -1, 0},
                               {0, 0, 1},   {0, 0, -1}, {1, 1, 0},  {1, -1, 0}, {-1, 1, 0},
                               {-1, -1, 0}, {1, 0, 1},  {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
                               {0, 1, 1},   {0, 1, -1}, {0, -1, 1}, {0, -1, -1}};

static constexpr double w_lb19[19] = {1. / 3.,  1. / 18., 1. / 18., 1. / 18., 1. / 18.,
                                  1. / 18., 1. / 18., 1. / 36., 1. / 36., 1. / 36.,
                                  1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.,
                                  1. / 36., 1. / 36., 1. / 36., 1. / 36.};

static constexpr double Ng_lb19[19] = {1.,  3.,  3.,        3.,        9. / 2.,  9. / 2.,  9. / 2.,
                                   9.,  9.,  9.,        27. / 2.,  27. / 2., 27. / 2., 18.,
                                   18., 18., 162. / 7., 126. / 5., 30.};

// clang-format off
static constexpr double mg_lb19[19][19] = {
  {    1.,     1.,     1.,     1.,     1.,     1.,     1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.},
  {    0.,     1.,     0.,    -1.,     0.,     0.,     0.,    1.,    1.,   -1.,   -1.,    1.,    1.,   -1.,   -1.,    0.,    0.,    0.,    0.},
  {    0.,     0.,     1.,     0.,    -1.,     0.,     0.,    1.,   -1.,    1.,   -1.,    0.,    0.,    0.,    0.,    1.,    1.,   -1.,   -1.},
  {    0.,     0.,     0.,     0.,     0.,     1.,    -1.,    0.,    0.,    0.,    0.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.},
  {-1./3.,  2./3., -1./3.,  2./3., -1./3., -1./3., -1./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3.,-1./3.,-1./3.,-1./3.,-1./3.},
  {-1./3., -1./3.,  2./3., -1./3.,  2./3., -1./3., -1./3., 2./3., 2./3., 2./3., 2./3.,-1./3.,-1./3.,-1./3.,-1./3., 2./3., 2./3., 2./3., 2./3.},
  {-1./3., -1./3., -1./3., -1./3., -1./3.,  2./3.,  2./3.,-1./3.,-1./3.,-1./3.,-1./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3.},
  {    0.,     0.,     0.,     0.,     0.,     0.,     0.,    1.,   -1.,   -1.,    1.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.},
  {    0.,     0.,     0.,     0.,     0.,     0.,     0.,    0.,    0.,    0.,    0.,    1.,   -1.,   -1.,    1.,    0.,    0.,    0.,    0.},
  {    0.,     0.,     0.,     0.,     0.,     0.,     0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    1.,   -1.,   -1.,    1.},
  {    0., -1./3.,     0.,  1./3.,     0.,     0.,     0., 2./3., 2./3.,-2./3.,-2./3.,-1./3.,-1./3., 1./3., 1./3.,    0.,    0.,    0.,    0.},
  {    0.,     0., -1./3.,     0.,  1./3.,     0.,     0., 2./3.,-2./3., 2./3.,-2./3.,    0.,    0.,    0.,    0.,-1./3.,-1./3., 1./3., 1./3.},
  {    0.,     0.,     0.,     0.,     0., -1./3.,  1./3.,    0.,    0.,    0.,    0., 2./3.,-2./3., 2./3.,-2./3.,-1./3., 1./3.,-1./3., 1./3.},
  {    0.,   -0.5,     0.,    0.5,     0.,     0.,     0.,    0.,    0.,    0.,    0.,   0.5,   0.5,  -0.5,  -0.5,    0.,    0.,    0.,    0.},
  {    0.,     0.,     0.,     0.,     0.,   -0.5,    0.5,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,   0.5,  -0.5,   0.5,  -0.5},
  {    0.,     0.,   -0.5,     0.,    0.5,     0.,     0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,   0.5,   0.5,  -0.5,  -0.5},
  {1./18.,-5./18.,-5./18.,-5./18.,-5./18.,  2./9.,  2./9.,7./18.,7./18.,7./18.,7./18.,-1./9.,-1./9.,-1./9.,-1./9.,-1./9.,-1./9.,-1./9.,-1./9.},
  {1./14.,-5./14.,  1./7.,-5./14.,  1./7.,-3./14.,-3./14.,    0.,    0.,    0.,    0.,5./14.,5./14.,5./14.,5./14.,-1./7.,-1./7.,-1./7.,-1./7.},
  {1./10.,     0.,-3./10.,     0.,-3./10.,-3./10.,-3./10.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,3./10.,3./10.,3./10.,3./10.}
};
// clang-format on

#endif

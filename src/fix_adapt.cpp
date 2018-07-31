/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include "fix_adapt.h"
#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "dihedral.h"
#include "update.h"
#include "group.h"
#include "modify.h"
#include "force.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "kspace.h"
#include "fix_store.h"
#include "input.h"
#include "variable.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{PAIR,KSPACE,ATOM,BOND,ANGLE,DIHEDRAL};
enum{DIAMETER,CHARGE};

/* ---------------------------------------------------------------------- */

FixAdapt::FixAdapt(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),
nadapt(0), id_fix_diam(NULL), id_fix_chg(NULL), adapt(NULL)
{
  if (narg < 5) error->all(FLERR,"Illegal fix adapt command");
  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery < 0) error->all(FLERR,"Illegal fix adapt command");

  dynamic_group_allow = 1;
  create_attribute = 1;

  // count # of adaptations

  nadapt = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+6 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 6;
    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"atom") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"bond") == 0 ){
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else if (strcmp(arg[iarg],"angle") == 0 ){
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else if (strcmp(arg[iarg],"dihedral") == 0 ){
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else break;
  }

  if (nadapt == 0) error->all(FLERR,"Illegal fix adapt command");
  adapt = new Adapt[nadapt];

  // parse keywords

  nadapt = 0;
  diamflag = 0;
  chgflag = 0;

  iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+6 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = PAIR;
      int n = strlen(arg[iarg+1]) + 1;
      adapt[nadapt].pstyle = new char[n];
      strcpy(adapt[nadapt].pstyle,arg[iarg+1]);
      n = strlen(arg[iarg+2]) + 1;
      adapt[nadapt].pparam = new char[n];
      adapt[nadapt].pair = NULL;
      strcpy(adapt[nadapt].pparam,arg[iarg+2]);
      force->bounds(FLERR,arg[iarg+3],atom->ntypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi);
      force->bounds(FLERR,arg[iarg+4],atom->ntypes,
                    adapt[nadapt].jlo,adapt[nadapt].jhi);
      if (strstr(arg[iarg+5],"v_") == arg[iarg+5]) {
        n = strlen(&arg[iarg+5][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+5][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 6;
    } else if (strcmp(arg[iarg],"bond") == 0 ){
      if (iarg+5 > narg) error->all(FLERR, "Illegal fix adapt command");
      adapt[nadapt].which = BOND;
      int n = strlen(arg[iarg+1]) + 1;
      adapt[nadapt].bstyle = new char[n];
      strcpy(adapt[nadapt].bstyle,arg[iarg+1]);
      n = strlen(arg[iarg+2]) + 1;
      adapt[nadapt].bparam = new char[n];
      adapt[nadapt].bond = NULL;
      strcpy(adapt[nadapt].bparam,arg[iarg+2]);
      force->bounds(FLERR,arg[iarg+3],atom->nbondtypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi);
      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
        n = strlen(&arg[iarg+4][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+4][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else if (strcmp(arg[iarg],"angle") == 0 ){
      if (iarg+5 > narg) error->all(FLERR, "Illegal fix adapt command");
      adapt[nadapt].which = ANGLE;
      int n = strlen(arg[iarg+1]) + 1;
      adapt[nadapt].vastyle = new char[n];
      strcpy(adapt[nadapt].vastyle,arg[iarg+1]);
      n = strlen(arg[iarg+2]) + 1;
      adapt[nadapt].vaparam = new char[n];
      adapt[nadapt].vangle = NULL;
      strcpy(adapt[nadapt].vaparam,arg[iarg+2]);
      force->bounds(FLERR,arg[iarg+3],atom->nangletypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi);
      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) { 
        n = strlen(&arg[iarg+4][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+4][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else if (strcmp(arg[iarg],"dihedral") == 0 ){
      if (iarg+5 > narg) error->all(FLERR, "Illegal fix adapt command");
      adapt[nadapt].which = DIHEDRAL;
      int n = strlen(arg[iarg+1]) + 1;
      adapt[nadapt].dstyle = new char[n];
      strcpy(adapt[nadapt].dstyle,arg[iarg+1]);
      n = strlen(arg[iarg+2]) + 1;
      adapt[nadapt].dparam = new char[n];
      adapt[nadapt].dihedral = NULL;
      strcpy(adapt[nadapt].dparam,arg[iarg+2]);
      force->bounds(FLERR,arg[iarg+3],atom->ndihedraltypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi);
      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
        n = strlen(&arg[iarg+4][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+4][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = KSPACE;
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"atom") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = ATOM;
      if (strcmp(arg[iarg+1],"diameter") == 0) {
        adapt[nadapt].aparam = DIAMETER;
        diamflag = 1;
      } else if (strcmp(arg[iarg+1],"charge") == 0) {
        adapt[nadapt].aparam = CHARGE;
        chgflag = 1;
      } else error->all(FLERR,"Illegal fix adapt command");
      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
        int n = strlen(&arg[iarg+2][2]) + 1;
        adapt[nadapt].var = new char[n];
        strcpy(adapt[nadapt].var,&arg[iarg+2][2]);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 3;
    } else break;
  }

  // optional keywords

  resetflag = 0;
  scaleflag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"reset") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      if (strcmp(arg[iarg+1],"no") == 0) resetflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) resetflag = 1;
      else error->all(FLERR,"Illegal fix adapt command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      if (strcmp(arg[iarg+1],"no") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix adapt command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix adapt command");
  }

  // allocate pair style arrays

  int n = atom->ntypes;
  for (int m = 0; m < nadapt; m++)
    if (adapt[m].which == PAIR)
      memory->create(adapt[m].array_orig,n+1,n+1,"adapt:array_orig");

  // allocate bond style arrays:

  n = atom->nbondtypes;
  for (int m = 0; m < nadapt; ++m)
    if (adapt[m].which == BOND)
      memory->create(adapt[m].vector_orig,n+1,"adapt:vector_orig");

  // allocate angle style arrays:

  n = atom->nangletypes;
  for (int m = 0; m < nadapt; ++m)
    if (adapt[m].which == ANGLE)
      memory->create(adapt[m].vector_orig,n+1,"adapt:vector_orig");

  // allocate dihedral style arrays:

  n = atom->ndihedraltypes;
  for (int m = 0; m < nadapt; ++m)
    if (adapt[m].which == DIHEDRAL)
      memory->create(adapt[m].vector_orig,n+1,"adapt:vector_orig");
}

/* ---------------------------------------------------------------------- */

FixAdapt::~FixAdapt()
{
  for (int m = 0; m < nadapt; m++) {
    delete [] adapt[m].var;
    if (adapt[m].which == PAIR) {
      delete [] adapt[m].pstyle;
      delete [] adapt[m].pparam;
      memory->destroy(adapt[m].array_orig);
    } else if (adapt[m].which == BOND) {
      delete [] adapt[m].bstyle;
      delete [] adapt[m].bparam;
      memory->destroy(adapt[m].vector_orig);
    } else if (adapt[m].which == ANGLE) {
      delete [] adapt[m].vastyle;
      delete [] adapt[m].vaparam;
      memory->destroy(adapt[m].vector_orig);
    } else if (adapt[m].which == DIHEDRAL) {
      delete [] adapt[m].dstyle;
      delete [] adapt[m].dparam;
      memory->destroy(adapt[m].vector_orig);
    }
  }
  delete [] adapt;

  // check nfix in case all fixes have already been deleted

  if (id_fix_diam && modify->nfix) modify->delete_fix(id_fix_diam);
  if (id_fix_chg && modify->nfix) modify->delete_fix(id_fix_chg);
  delete [] id_fix_diam;
  delete [] id_fix_chg;
}

/* ---------------------------------------------------------------------- */

int FixAdapt::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_RUN;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ----------------------------------------------------------------------
   if need to restore per-atom quantities, create new fix STORE styles
------------------------------------------------------------------------- */

void FixAdapt::post_constructor()
{
  if (!resetflag) return;
  if (!diamflag && !chgflag) return;

  // new id = fix-ID + FIX_STORE_ATTRIBUTE
  // new fix group = group for this fix

  id_fix_diam = NULL;
  id_fix_chg = NULL;

  char **newarg = new char*[6];
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "STORE";
  newarg[3] = (char *) "peratom";
  newarg[4] = (char *) "1";
  newarg[5] = (char *) "1";

  if (diamflag) {
    int n = strlen(id) + strlen("_FIX_STORE_DIAM") + 1;
    id_fix_diam = new char[n];
    strcpy(id_fix_diam,id);
    strcat(id_fix_diam,"_FIX_STORE_DIAM");
    newarg[0] = id_fix_diam;
    modify->add_fix(6,newarg);
    fix_diam = (FixStore *) modify->fix[modify->nfix-1];

    if (fix_diam->restart_reset) fix_diam->restart_reset = 0;
    else {
      double *vec = fix_diam->vstore;
      double *radius = atom->radius;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = radius[i];
        else vec[i] = 0.0;
      }
    }
  }

  if (chgflag) {
    int n = strlen(id) + strlen("_FIX_STORE_CHG") + 1;
    id_fix_chg = new char[n];
    strcpy(id_fix_chg,id);
    strcat(id_fix_chg,"_FIX_STORE_CHG");
    newarg[0] = id_fix_chg;
    modify->add_fix(6,newarg);
    fix_chg = (FixStore *) modify->fix[modify->nfix-1];

    if (fix_chg->restart_reset) fix_chg->restart_reset = 0;
    else {
      double *vec = fix_chg->vstore;
      double *q = atom->q;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = q[i];
        else vec[i] = 0.0;
      }
    }
  }

  delete [] newarg;
}

/* ---------------------------------------------------------------------- */

void FixAdapt::init()
{
  int i,j;

  // allow a dynamic group only if ATOM attribute not used

  if (group->dynamic[igroup])
    for (int i = 0; i < nadapt; i++)
      if (adapt[i].which == ATOM)
        error->all(FLERR,"Cannot use dynamic group with fix adapt atom");

  // setup and error checks

  anypair = 0;
  anybond = 0;
  anyangle = 0;
  anydihedral = 0;

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];

    ad->ivar = input->variable->find(ad->var);
    if (ad->ivar < 0)
      error->all(FLERR,"Variable name for fix adapt does not exist");
    if (!input->variable->equalstyle(ad->ivar))
      error->all(FLERR,"Variable for fix adapt is invalid style");

    if (ad->which == PAIR) {
      anypair = 1;
      ad->pair = NULL;

      // if ad->pstyle has trailing sub-style annotation ":N",
      //   strip it for pstyle arg to pair_match() and set nsub = N
      // this should work for appended suffixes as well

      int n = strlen(ad->pstyle) + 1;
      char *pstyle = new char[n];
      strcpy(pstyle,ad->pstyle);

      char *cptr;
      int nsub = 0;
      if ((cptr = strchr(pstyle,':'))) {
        *cptr = '\0';
        nsub = force->inumeric(FLERR,cptr+1);
      }

      if (lmp->suffix_enable) {
        int len = 2 + strlen(pstyle) + strlen(lmp->suffix);
        char *psuffix = new char[len];
        strcpy(psuffix,pstyle);
        strcat(psuffix,"/");
        strcat(psuffix,lmp->suffix);
        ad->pair = force->pair_match(psuffix,1,nsub);
        delete[] psuffix;
      }
      if (ad->pair == NULL) ad->pair = force->pair_match(pstyle,1,nsub);
      if (ad->pair == NULL)
        error->all(FLERR,"Fix adapt pair style does not exist");

      void *ptr = ad->pair->extract(ad->pparam,ad->pdim);
      if (ptr == NULL)
        error->all(FLERR,"Fix adapt pair style param not supported");

      // for pair styles only parameters that are 2-d arrays in atom types or
      // scalars are supported

      if (ad->pdim != 2 && ad->pdim != 0)
        error->all(FLERR,"Fix adapt pair style param is not compatible");

      if (ad->pdim == 2) ad->array = (double **) ptr;
      if (ad->pdim == 0) ad->scalar = (double *) ptr;

      // if pair hybrid, test that ilo,ihi,jlo,jhi are valid for sub-style

      if (strcmp(force->pair_style,"hybrid") == 0 ||
          strcmp(force->pair_style,"hybrid/overlay") == 0) {
        PairHybrid *pair = (PairHybrid *) force->pair;
        for (i = ad->ilo; i <= ad->ihi; i++)
          for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
            if (!pair->check_ijtype(i,j,pstyle))
              error->all(FLERR,"Fix adapt type pair range is not valid for "
                         "pair hybrid sub-style");
      }

      delete [] pstyle;
    } else if (ad->which == BOND){
      ad->bond = NULL;
      anybond = 1;

      int n = strlen(ad->bstyle) + 1;
      char *bstyle = new char[n];
      strcpy(bstyle,ad->bstyle);

      if (lmp->suffix_enable) {
        int len = 2 + strlen(bstyle) + strlen(lmp->suffix);
        char *bsuffix = new char[len];
        strcpy(bsuffix,bstyle);
        strcat(bsuffix,"/");
        strcat(bsuffix,lmp->suffix);
        ad->bond = force->bond_match(bsuffix);
        delete [] bsuffix;
      }
      if (ad->bond == NULL) ad->bond = force->bond_match(bstyle);
      if (ad->bond == NULL )
        error->all(FLERR,"Fix adapt bond style does not exist");

      void *ptr = ad->bond->extract(ad->bparam,ad->bdim);

      if (ptr == NULL)
        error->all(FLERR,"Fix adapt bond style param not supported");

      // for bond styles, use a vector

      if (ad->bdim == 1) ad->vector = (double *) ptr;

      if (strcmp(force->bond_style,"hybrid") == 0 ||
          strcmp(force->bond_style,"hybrid_overlay") == 0)
        error->all(FLERR,"Fix adapt does not support bond_style hybrid");

      delete [] bstyle;
        
    } else if (ad->which == ANGLE){
      ad->vangle = NULL;
      anyangle = 1;
      
      int n = strlen(ad->vastyle) + 1;
      char *vastyle = new char[n];
      strcpy(vastyle,ad->vastyle);

      if (lmp->suffix_enable) {
        int len = 2 + strlen(vastyle) + strlen(lmp->suffix);
        char *vasuffix = new char[len];
        strcpy(vasuffix,vastyle);
        strcat(vasuffix,"/");
        strcat(vasuffix,lmp->suffix);
        ad->vangle = force->angle_match(vasuffix);
        delete [] vasuffix;
      }
      if (ad->vangle == NULL) ad->vangle = force->angle_match(vastyle);
      if (ad->vangle == NULL )
        error->all(FLERR,"Fix adapt angle style does not exist");

      void *ptr = ad->vangle->extract(ad->vaparam,ad->vadim);
      
      if (ptr == NULL)
        error->all(FLERR,"Fix adapt angle style param not supported");

      // for angle styles, use a vector

      if (ad->vadim == 1) ad->vector = (double *) ptr;

      if (strcmp(force->angle_style,"hybrid") == 0 ||
          strcmp(force->angle_style,"hybrid_overlay") == 0)
        error->all(FLERR,"Fix adapt does not support angle_style hybrid");

      delete [] vastyle;
        
    } else if (ad->which == DIHEDRAL){
      ad->dihedral = NULL;
      anydihedral = 1;
      
      int n = strlen(ad->dstyle) + 1;
      char *dstyle = new char[n];
      strcpy(dstyle,ad->dstyle);

      if (lmp->suffix_enable) {
        int len = 2 + strlen(dstyle) + strlen(lmp->suffix);
        char *dsuffix = new char[len];
        strcpy(dsuffix,dstyle);
        strcat(dsuffix,"/");
        strcat(dsuffix,lmp->suffix);
        ad->dihedral = force->dihedral_match(dsuffix);
        delete [] dsuffix;
      }
      if (ad->dihedral == NULL) ad->dihedral = force->dihedral_match(dstyle);
      if (ad->dihedral == NULL )
        error->all(FLERR,"Fix adapt dihedral style does not exist");

      void *ptr = ad->dihedral->extract(ad->dparam,ad->ddim);
      
      if (ptr == NULL)
        error->all(FLERR,"Fix adapt dihedral style param not supported");

      // for dihedral styles, use a vector

      if (ad->ddim == 1) ad->vector = (double *) ptr;

      if (strcmp(force->dihedral_style,"hybrid") == 0 ||
          strcmp(force->dihedral_style,"hybrid_overlay") == 0)
        error->all(FLERR,"Fix adapt does not support dihedral_style hybrid");

      delete [] dstyle;
        
    } else if (ad->which == KSPACE) {
      if (force->kspace == NULL)
        error->all(FLERR,"Fix adapt kspace style does not exist");
      kspace_scale = (double *) force->kspace->extract("scale");

    } else if (ad->which == ATOM) {
      if (ad->aparam == DIAMETER) {
        if (!atom->radius_flag)
          error->all(FLERR,"Fix adapt requires atom attribute diameter");
      }
      if (ad->aparam == CHARGE) {
        if (!atom->q_flag)
          error->all(FLERR,"Fix adapt requires atom attribute charge");
      }
    }
  }

  // make copy of original pair/bond/angle/dihedral array values

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    if (ad->which == PAIR && ad->pdim == 2) {
      for (i = ad->ilo; i <= ad->ihi; i++)
        for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
          ad->array_orig[i][j] = ad->array[i][j];
    } else if (ad->which == PAIR && ad->pdim == 0){
      ad->scalar_orig = *ad->scalar;

    } else if (ad->which == BOND && ad->bdim == 1){
      for (i = ad->ilo; i <= ad->ihi; ++i )
        ad->vector_orig[i] = ad->vector[i];
    } else if (ad->which == ANGLE && ad->vadim == 1){
      for (i = ad->ilo; i <= ad->ihi; ++i )
        ad->vector_orig[i] = ad->vector[i];
    } else if (ad->which == DIHEDRAL && ad->ddim == 1){
      for (i = ad->ilo; i <= ad->ihi; ++i )
        ad->vector_orig[i] = ad->vector[i];
    }

  }

  // fixes that store initial per-atom values

  if (id_fix_diam) {
    int ifix = modify->find_fix(id_fix_diam);
    if (ifix < 0) error->all(FLERR,"Could not find fix adapt storage fix ID");
    fix_diam = (FixStore *) modify->fix[ifix];
  }
  if (id_fix_chg) {
    int ifix = modify->find_fix(id_fix_chg);
    if (ifix < 0) error->all(FLERR,"Could not find fix adapt storage fix ID");
    fix_chg = (FixStore *) modify->fix[ifix];
  }

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixAdapt::setup_pre_force(int vflag)
{
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixAdapt::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdapt::pre_force(int vflag)
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixAdapt::pre_force_respa(int vflag, int ilevel, int)
{
  if (ilevel < nlevels_respa-1) return;
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdapt::post_run()
{
  if (resetflag) restore_settings();
}

/* ----------------------------------------------------------------------
   change pair,kspace,atom parameters based on variable evaluation
------------------------------------------------------------------------- */

void FixAdapt::change_settings()
{
  int i,j;

  // variable evaluation may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    double value = input->variable->compute_equal(ad->ivar);

    // set global scalar or type pair array values

    if (ad->which == PAIR) {
      if (ad->pdim == 0) {
        if (scaleflag) *ad->scalar = value * ad->scalar_orig;
        else *ad->scalar = value;
      } else if (ad->pdim == 2) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value*ad->array_orig[i][j];
        else
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value;
      }

    // set bond/angle/dihedral type array values:
      
    } else if (ad->which == BOND) {
      if (ad->bdim == 1){
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
      }
    } else if (ad->which == ANGLE) {
      if (ad->vadim == 1){
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else {
            if (strcmp(ad->vaparam, "theta") == 0)
                value *= MathConst::MY_PI/180.0; //convert from degrees to radians
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
        }
      }
    } else if (ad->which == DIHEDRAL) {
      if (ad->ddim == 1){
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else {
            if (strcmp(ad->dstyle, "opls") == 0)
                value *= 0.5; //for opls include 0.5 factor
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
        }
      }
      
    // set kspace scale factor

    } else if (ad->which == KSPACE) {
      *kspace_scale = value;

    // set per atom values, also make changes for ghost atoms

    } else if (ad->which == ATOM) {

      // reset radius from diameter
      // also scale rmass to new value

      if (ad->aparam == DIAMETER) {
        int mflag = 0;
        if (atom->rmass_flag) mflag = 1;
        double density;

        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        if (mflag == 0) {
          for (i = 0; i < nall; i++)
            if (mask[i] & groupbit)
              radius[i] = 0.5*value;
        } else {
          for (i = 0; i < nall; i++)
            if (mask[i] & groupbit) {
              density = rmass[i] / (4.0*MY_PI/3.0 *
                                    radius[i]*radius[i]*radius[i]);
              radius[i] = 0.5*value;
              rmass[i] = 4.0*MY_PI/3.0 *
                radius[i]*radius[i]*radius[i] * density;
            }
        }
      } else if (ad->aparam == CHARGE) {
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        for (i = 0; i < nall; i++)
          if (mask[i] & groupbit) q[i] = value;
      }
    }
  }

  modify->addstep_compute(update->ntimestep + nevery);

  // re-initialize pair styles if any PAIR settings were changed
  // ditto for bond styles if any BOND setitings were changes
  // this resets other coeffs that may depend on changed values,
  //   and also offset and tail corrections

  if (anypair) {
    for (int m = 0; m < nadapt; m++) {
      Adapt *ad = &adapt[m];
      if (ad->which == PAIR) {
        ad->pair->reinit();
      }
    }
  }
  if (anybond) {
    for (int m = 0; m < nadapt; ++m ) {
      Adapt *ad = &adapt[m];
      if (ad->which == BOND) {
        ad->bond->reinit();
      }
    }
  }
  if (anyangle) {
    for (int m = 0; m < nadapt; ++m ) {
      Adapt *ad = &adapt[m];
      if (ad->which == ANGLE) {
        ad->vangle->reinit();
      }
    }
  }
  if (anydihedral) {
    for (int m = 0; m < nadapt; ++m ) {
      Adapt *ad = &adapt[m];
      if (ad->which == DIHEDRAL) {
        ad->dihedral->reinit();
      }
    }
  }

  // reset KSpace charges if charges have changed

  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   restore pair,kspace,atom parameters to original values
------------------------------------------------------------------------- */

void FixAdapt::restore_settings()
{
  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    if (ad->which == PAIR) {
      if (ad->pdim == 0) *ad->scalar = ad->scalar_orig;
      else if (ad->pdim == 2) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          for (int j = MAX(ad->jlo,i); j <= ad->jhi; j++)
            ad->array[i][j] = ad->array_orig[i][j];
      }

    } else if (ad->which == BOND || ad->which == ANGLE ||
            ad->which == DIHEDRAL) {
      if (ad->pdim == 1) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          ad->vector[i] = ad->vector_orig[i];
      }

    } else if (ad->which == KSPACE) {
      *kspace_scale = 1.0;

    } else if (ad->which == ATOM) {
      if (diamflag) {
        double density;

        double *vec = fix_diam->vstore;
        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) {
            density = rmass[i] / (4.0*MY_PI/3.0 *
                                  radius[i]*radius[i]*radius[i]);
            radius[i] = vec[i];
            rmass[i] = 4.0*MY_PI/3.0 * radius[i]*radius[i]*radius[i] * density;
          }
      }
      if (chgflag) {
        double *vec = fix_chg->vstore;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) q[i] = vec[i];
      }
    }
  }

  if (anypair) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void FixAdapt::set_arrays(int i)
{
  if (fix_diam) fix_diam->vstore[i] = atom->radius[i];
  if (fix_chg) fix_chg->vstore[i] = atom->q[i];
}

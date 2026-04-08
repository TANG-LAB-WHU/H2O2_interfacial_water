/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Kristof Bal (University of Antwerp)
   Based on: Bal & Neyts, JCTC 2015, 10.1021/acs.jctc.5b00197
   "Merging Metadynamics into Hyperdynamics: Accelerated Molecular
    Simulations Reaching Time Scales from Microseconds to Seconds"
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(timeboost, FixTimeboost);
// clang-format on
#else

#ifndef LMP_FIX_TIMEBOOST_H
#define LMP_FIX_TIMEBOOST_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTimeboost : public Fix {
 public:
  FixTimeboost(class LAMMPS *, int, char **);
  ~FixTimeboost() override;
  int setmask() override;
  void init() override;
  void end_of_step() override;
  double compute_scalar() override;

 private:
  char *id_fix;          // ID of the fix that provides bias energy
  class Fix *fix_energy; // Pointer to the fix providing bias energy
  double tset;           // Target temperature (K)
  double boost_sum;      // Accumulated boost time
};

}    // namespace LAMMPS_NS

#endif
#endif

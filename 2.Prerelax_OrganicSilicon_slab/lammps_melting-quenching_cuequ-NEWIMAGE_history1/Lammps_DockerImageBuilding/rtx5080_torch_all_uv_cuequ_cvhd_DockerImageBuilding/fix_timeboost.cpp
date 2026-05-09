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

/* ----------------------------------------------------------------------
   Contributing author: Kristof Bal (University of Antwerp)
   Based on: Bal & Neyts, JCTC 2015, 10.1021/acs.jctc.5b00197
   "Merging Metadynamics into Hyperdynamics: Accelerated Molecular
    Simulations Reaching Time Scales from Microseconds to Seconds"
------------------------------------------------------------------------- */

#include "fix_timeboost.h"
#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixTimeboost::FixTimeboost(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  id_fix(nullptr),
  fix_energy(nullptr)
{
  if (narg != 5)
    error->all(FLERR, "Illegal fix timeboost command: expected 5 arguments");

  // Parse arguments: fix ID group timeboost fix_id temperature
  // arg[0] = fix ID
  // arg[1] = group ID
  // arg[2] = "timeboost"
  // arg[3] = ID of the fix that provides bias energy (e.g., plumed or colvars)
  // arg[4] = target temperature (K) for boost factor calculation

  int n = strlen(arg[3]) + 1;
  id_fix = new char[n];
  strcpy(id_fix, arg[3]);

  tset = utils::numeric(FLERR, arg[4], false, lmp);
  if (tset <= 0.0)
    error->all(FLERR, "Fix timeboost temperature must be positive");

  // Initialize accumulated boost
  boost_sum = 0.0;

  // Enable energy extraction from this fix (for thermo output)
  scalar_flag = 1;
  extscalar = 1;
  global_freq = 1;
}

/* ---------------------------------------------------------------------- */

FixTimeboost::~FixTimeboost()
{
  delete[] id_fix;
}

/* ---------------------------------------------------------------------- */

int FixTimeboost::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTimeboost::init()
{
  // Find the fix that provides the bias energy
  fix_energy = modify->get_fix_by_id(id_fix);
  if (!fix_energy)
    error->all(FLERR, "Fix timeboost could not find fix ID: {}", id_fix);

  // Verify the fix can provide a scalar energy value
  if (fix_energy->scalar_flag == 0)
    error->all(FLERR, "Fix timeboost requires fix that computes scalar energy");
}

/* ---------------------------------------------------------------------- */

void FixTimeboost::end_of_step()
{
  // Get the bias energy from the specified fix (e.g., PLUMED or Colvars)
  double eboost = fix_energy->compute_scalar();

  // Boltzmann constant in LAMMPS energy units per Kelvin
  double boltz = force->boltz;

  // Calculate the boost factor: B = exp(V_bias / (k_B * T))
  // This is the core hyperdynamics formula (Voter, 1997)
  double boost = exp(eboost / (boltz * tset));

  // Accumulate the total boosted time
  double dt = update->dt;
  boost_sum += (boost - 1.0) * dt;

  // Add the boosted time to LAMMPS accumulated time (update->atime)
  // This makes the boosted time visible in thermo output and dump files
  update->atime += (boost - 1.0) * dt;
}

/* ---------------------------------------------------------------------- */

double FixTimeboost::compute_scalar()
{
  // Return the accumulated boost time for thermo output
  // User can access this via f_timeboost
  return boost_sum;
}

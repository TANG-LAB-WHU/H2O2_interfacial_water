/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2020 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

/* ----------------------------------------------------------------------
   BondRotate CV for CVHD (Collective Variable Hyperdynamics)
   Ported from Colvars library implementation by Kristof Bal
   Original reference: Bal & Neyts, JCTC 2015, 10.1021/acs.jctc.5b00197
   
   This CV monitors both bond rotation AND bond breaking simultaneously.
   It combines the rotation displacement of bond endpoints from reference
   positions with the bond stretching (bondbreak) contribution.
---------------------------------------------------------------------- */

#include "core/Colvar.h"
#include "core/ActionRegister.h"

#include <string>
#include <cmath>
#include <vector>

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR BONDROTATE
/*
Calculate the bond rotation collective variable for CVHD.

This CV monitors both bond rotation AND bond breaking events simultaneously.
For each bond pair, it computes:
1. Rotation displacement: how much the bond endpoints have moved perpendicular to the bond axis
2. Bond stretching: how much the bond has stretched (using bondbreak-like switching)

Both contributions are aggregated using a p-norm. The CV uses a smooth
switching function to map distortions to [0,1].

\par Examples

Calculate the bond rotation CV for atoms in two groups:
\plumedfile
BONDROTATE GROUPA=1-10 GROUPB=11-20 RMIN=0.1 RMAX=0.2 DMAX=0.05 RCUT=0.15 P=12 LABEL=br
PRINT ARG=br FILE=bondrotate.out
\endplumedfile

*/
//+ENDPLUMEDOC

class BondRotate : public Colvar {
private:
  bool pbc;
  bool oneGroup;
  bool reBuildReflist;
  bool reset_ref;
  double rmin;          // Minimum bond distance for switching
  double rmax;          // Maximum bond distance (broken bond)
  double dmax;          // Maximum rotation displacement
  double rcut;          // Cutoff for bond detection
  double power;         // P-norm exponent
  double reset_maxdist; // Threshold for reset
  unsigned reset_time;  // Steps to wait before reset
  unsigned reset_wait;  // Current wait counter
  unsigned offset;      // Event counter

  vector<unsigned> pairList1;  // Indices into group A
  vector<unsigned> pairList2;  // Indices into group B
  vector<Vector> oldList1;     // Reference positions (endpoint 1)
  vector<Vector> oldList2;     // Reference positions (endpoint 2)

  vector<AtomNumber> groupA;
  vector<AtomNumber> groupB;

  void buildBondlist();
  double rotationSwitchingFunction(double dmax_val, double p, double prefactor,
                                   unsigned i, unsigned j, Vector& old1, Vector& old2,
                                   vector<Vector>& deriv, Tensor& virial);
  double stretchSwitchingFunction(double rmin_val, double rmax_val, double p, double prefactor,
                                  unsigned i, unsigned j, vector<Vector>& deriv, Tensor& virial);

public:
  explicit BondRotate(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(BondRotate, "BONDROTATE")

void BondRotate::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms", "GROUPA", "First list of atoms");
  keys.add("atoms", "GROUPB", "Second list of atoms (if empty, only pairs within GROUPA are considered)");
  keys.add("compulsory", "RMIN", "0.1", "Minimum bond distance for switching function (nm)");
  keys.add("compulsory", "RMAX", "0.2", "Maximum bond distance (bond considered broken) (nm)");
  keys.add("compulsory", "DMAX", "0.05", "Maximum rotation displacement (nm)");
  keys.add("compulsory", "RCUT", "0.15", "Cutoff for detecting bonds at first step (nm)");
  keys.add("compulsory", "P", "12", "The p parameter for computing the p-norm");
  keys.addFlag("RESET_REF", false, "Reset the reference list when CV exceeds threshold");
  keys.add("optional", "RESET_MAXDIST", "Threshold for resetting (if RESET_REF is enabled)");
  keys.add("optional", "RESET_TIME", "Number of steps to wait before resetting");
}

BondRotate::BondRotate(const ActionOptions& ao) :
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  oneGroup(false),
  reBuildReflist(true),
  reset_ref(false),
  reset_time(100),
  reset_wait(0),
  offset(0)
{
  parseAtomList("GROUPA", groupA);
  parseAtomList("GROUPB", groupB);

  if (groupA.empty()) error("GROUPA must be specified");
  if (groupB.empty()) {
    oneGroup = true;
    groupB = groupA;
  }

  parse("RMIN", rmin);
  parse("RMAX", rmax);
  parse("DMAX", dmax);
  parse("RCUT", rcut);
  parse("P", power);

  parseFlag("RESET_REF", reset_ref);
  parse("RESET_MAXDIST", reset_maxdist);
  parse("RESET_TIME", reset_time);

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  checkRead();

  // Request all atoms
  vector<AtomNumber> all_atoms = groupA;
  if (!oneGroup) {
    all_atoms.insert(all_atoms.end(), groupB.begin(), groupB.end());
  }
  requestAtoms(all_atoms);

  addValueWithDerivatives();
  setNotPeriodic();

  log.printf("  GROUPA has %zu atoms, GROUPB has %zu atoms\n", groupA.size(), groupB.size());
  log.printf("  rmin=%f rmax=%f dmax=%f rcut=%f power=%f\n", rmin, rmax, dmax, rcut, power);
}

void BondRotate::buildBondlist() {
  reBuildReflist = false;
  double cut2 = rcut * rcut;

  pairList1.clear();
  pairList2.clear();
  oldList1.clear();
  oldList2.clear();

  unsigned numA = groupA.size();
  unsigned numB = groupB.size();

  if (oneGroup) {
    for (unsigned i = 0; i < numA - 1; i++) {
      for (unsigned j = i + 1; j < numA; j++) {
        Vector distance;
        if (pbc) {
          distance = pbcDistance(getPosition(i), getPosition(j));
        } else {
          distance = delta(getPosition(i), getPosition(j));
        }
        double r2 = distance.modulo2();
        if (r2 < cut2) {
          pairList1.push_back(i);
          pairList2.push_back(j);
          // Store reference positions relative to bond center
          Vector new1 = -0.5 * distance;
          Vector new2 = 0.5 * distance;
          oldList1.push_back(new1);
          oldList2.push_back(new2);
        }
      }
    }
  } else {
    for (unsigned i = 0; i < numA; i++) {
      for (unsigned j = 0; j < numB; j++) {
        Vector distance;
        if (pbc) {
          distance = pbcDistance(getPosition(i), getPosition(numA + j));
        } else {
          distance = delta(getPosition(i), getPosition(numA + j));
        }
        double r2 = distance.modulo2();
        if (r2 < cut2) {
          pairList1.push_back(i);
          pairList2.push_back(numA + j);
          Vector new1 = -0.5 * distance;
          Vector new2 = 0.5 * distance;
          oldList1.push_back(new1);
          oldList2.push_back(new2);
        }
      }
    }
  }

  log.printf("  Found %zu bonds within cutoff\n", pairList1.size());
}

double BondRotate::rotationSwitchingFunction(double dmax_val, double p, double prefactor,
                                             unsigned i, unsigned j, Vector& old1, Vector& old2,
                                             vector<Vector>& deriv, Tensor& virial) {
  Vector distance;
  if (pbc) {
    distance = pbcDistance(getPosition(i), getPosition(j));
  } else {
    distance = delta(getPosition(i), getPosition(j));
  }

  Vector new1 = -0.5 * distance;
  Vector new2 = 0.5 * distance;

  // Compute displacement from reference (perpendicular component)
  Vector stretch_old = old2 - old1;
  Vector stretch_new = new2 - new1;
  double scale = stretch_old.modulo() / stretch_new.modulo();
  new1 *= scale;
  new2 *= scale;

  Vector diff1 = new1 - old1;
  Vector diff2 = new2 - old2;

  double dist1 = diff1.modulo();
  double dist2 = diff2.modulo();
  double dist = 0.5 * (dist1 + dist2);
  double xi = dist / dmax_val;

  // numerical stability...
  dist1 += 0.00001;
  dist2 += 0.00001;

  if (xi > 1.0) return 1.0;

  // Compute gradients
  if (prefactor != 0.0) {
    double dFdr = 0.5 * prefactor * pow(xi, p - 1.0) / dmax_val;
    Vector drdx1 = diff1 / dist1;
    Vector drdx2 = diff2 / dist2;
    deriv[i] += dFdr * drdx1;
    deriv[j] += dFdr * drdx2;
    Tensor vv(dFdr * drdx1, distance);
    virial -= vv;
  }

  return xi;
}

double BondRotate::stretchSwitchingFunction(double rmin_val, double rmax_val, double p, double prefactor,
                                            unsigned i, unsigned j, vector<Vector>& deriv, Tensor& virial) {
  Vector distance;
  if (pbc) {
    distance = pbcDistance(getPosition(i), getPosition(j));
  } else {
    distance = delta(getPosition(i), getPosition(j));
  }

  double dist = distance.modulo();

  if (dist > rmax_val) return 1.0;
  if (dist < rmin_val) return 0.0;

  double xi = (dist - rmin_val) / (rmax_val - rmin_val);

  // Compute gradients
  if (prefactor != 0.0) {
    double dFdr = prefactor * pow(xi, p - 1.0) / (rmax_val - rmin_val);
    Vector drdx = distance / dist;
    deriv[i] -= dFdr * drdx;
    deriv[j] += dFdr * drdx;
    Tensor vv(dFdr * drdx, distance);
    virial -= vv;
  }

  return xi;
}

void BondRotate::calculate() {
  if (pbc) makeWhole();

  // Build bond list if needed
  if (reBuildReflist) buildBondlist();

  vector<Vector> deriv(getNumberOfAtoms());
  Tensor virial;
  double pairsum = 0.0;

  // First pass: compute all switching values (no gradients)
  for (unsigned k = 0; k < pairList1.size(); k++) {
    unsigned i = pairList1[k];
    unsigned j = pairList2[k];

    // Rotation term
    double rot_xi = rotationSwitchingFunction(dmax, power, 0.0, i, j,
                                               oldList1[k], oldList2[k], deriv, virial);
    pairsum += pow(rot_xi, power);

    // Stretch term
    double stretch_xi = stretchSwitchingFunction(rmin, rmax, power, 0.0, i, j, deriv, virial);
    pairsum += pow(stretch_xi, power);
  }

  // Compute progress with cosine smoothing
  double progress = pow(pairsum, 1.0 / power);
  double value;
  double prefact = 0.0;

  if (progress < 1.0 && pairsum > pow(1e-10, power)) {
    value = 0.5 * (1.0 - cos(pi * progress * progress));
    prefact = pi * progress * sin(pi * progress * progress) * pow(pairsum, 1.0 / power - 1.0);
  } else if (progress >= 1.0) {
    value = 1.0;
  } else {
    value = 0.0;
  }

  // Add event offset
  value += 2.0 * static_cast<double>(offset);

  // Second pass: compute gradients
  if (prefact != 0.0) {
    for (unsigned k = 0; k < pairList1.size(); k++) {
      unsigned i = pairList1[k];
      unsigned j = pairList2[k];

      rotationSwitchingFunction(dmax, power, prefact, i, j,
                                oldList1[k], oldList2[k], deriv, virial);
      stretchSwitchingFunction(rmin, rmax, power, prefact, i, j, deriv, virial);
    }
  }

  // Set derivatives
  for (unsigned i = 0; i < getNumberOfAtoms(); i++) {
    setAtomsDerivatives(i, deriv[i]);
  }

  setBoxDerivatives(virial);
  setValue(value);

  // Bookkeeping for reset
  unsigned currstep = getStep();
  if (reset_ref) {
    if (value < reset_maxdist) {
      reset_wait = currstep;
    }
    if (currstep - reset_wait >= reset_time) {
      reset_wait = currstep;
      reBuildReflist = true;
      offset++;
    }
  }
}

}
}

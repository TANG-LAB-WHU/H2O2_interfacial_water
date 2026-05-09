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
   AngleSwitch CV for CVHD (Collective Variable Hyperdynamics)
   Ported from Colvars library implementation by Kristof Bal
   Original reference: Bal & Neyts, JCTC 2015, 10.1021/acs.jctc.5b00197
---------------------------------------------------------------------- */

#include "core/Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Torsion.h"

#include <string>
#include <cmath>
#include <vector>

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR ANGLESWITCH
/*
Calculate the angle switching collective variable for CVHD.

This CV monitors dihedral angle switching events. When a dihedral angle
deviates from its reference value beyond a specified threshold, the CV
value increases. Multiple dihedrals can be aggregated using a p-norm.

The CV uses a smooth switching function to map the angular deviation to [0,1]:
\f[
  \xi_i = \frac{|\theta_i - \theta_{ref}|}{w}
\f]

where \f$\theta_{ref}\f$ is the reference angle and \f$w\f$ is the width.
The final CV value is computed using a p-norm and cosine smoothing:
\f[
  s = \frac{1}{2}\left(1 - \cos\left(\pi \left(\sum_i \xi_i^p\right)^{2/p}\right)\right)
\f]

\par Examples

Calculate the angle switching CV for a set of dihedrals:
\plumedfile
ANGLESWITCH ATOMS1=1,2,3,4 ATOMS2=5,6,7,8 ANG_OFFSET=60 ANG_PERIOD=120 ANG_WIDTH=30 P=12 LABEL=as
PRINT ARG=as FILE=angleswitch.out
\endplumedfile

*/
//+ENDPLUMEDOC

class AngleSwitch : public Colvar {
private:
  bool pbc;
  bool reBuildReflist;
  bool reset_ref;
  double ang_offset;    // Reference angle offset (degrees)
  double ang_period;    // Angular period (degrees)
  double ang_width;     // Width for switching function (degrees)
  double power;         // P-norm exponent
  double reset_maxdist; // Threshold for reset
  unsigned reset_time;  // Steps to wait before reset
  unsigned reset_wait;  // Current wait counter
  unsigned offset;      // Event counter
  vector<double> angList;           // Reference angles for each dihedral
  vector<vector<AtomNumber>> atoms; // Atom groups for each dihedral

  double switching_function(double ang, double ang_ref, double ang_w, double p,
                           double prefactor, unsigned idx, vector<Vector>& deriv);
  void buildReflist();
  double computeTorsion(unsigned idx, vector<Vector>& deriv);

public:
  explicit AngleSwitch(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(AngleSwitch, "ANGLESWITCH")

void AngleSwitch::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("numbered", "ATOMS", "the four atoms defining each dihedral angle");
  keys.add("compulsory", "ANG_OFFSET", "60", "Reference angle offset in degrees");
  keys.add("compulsory", "ANG_PERIOD", "120", "Angular period in degrees");
  keys.add("compulsory", "ANG_WIDTH", "30", "Width of the switching function in degrees");
  keys.add("compulsory", "P", "12", "The p parameter for computing the p-norm");
  keys.addFlag("RESET_REF", false, "Reset the reference list when CV exceeds threshold");
  keys.add("optional", "RESET_MAXDIST", "Threshold for resetting (if RESET_REF is enabled)");
  keys.add("optional", "RESET_TIME", "Number of steps to wait before resetting");
}

AngleSwitch::AngleSwitch(const ActionOptions& ao) :
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  reBuildReflist(true),
  reset_ref(false),
  reset_time(100),
  reset_wait(0),
  offset(0)
{
  // Parse dihedral atom groups
  for (int i = 1;; ++i) {
    vector<AtomNumber> t;
    parseAtomList("ATOMS", i, t);
    if (t.empty()) break;
    if (t.size() != 4) error("ATOMS" + to_string(i) + " must specify exactly 4 atoms");
    atoms.push_back(t);
  }
  if (atoms.empty()) error("No ATOMS specified");

  parse("ANG_OFFSET", ang_offset);
  parse("ANG_PERIOD", ang_period);
  parse("ANG_WIDTH", ang_width);
  parse("P", power);

  parseFlag("RESET_REF", reset_ref);
  parse("RESET_MAXDIST", reset_maxdist);
  parse("RESET_TIME", reset_time);

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  checkRead();

  // Request all atoms
  vector<AtomNumber> all_atoms;
  for (auto& grp : atoms) {
    for (auto& a : grp) all_atoms.push_back(a);
  }
  requestAtoms(all_atoms);

  addValueWithDerivatives();
  setNotPeriodic();

  log.printf("  monitoring %zu dihedral angles\n", atoms.size());
  log.printf("  ang_offset=%f ang_period=%f ang_width=%f power=%f\n",
             ang_offset, ang_period, ang_width, power);
}

double AngleSwitch::computeTorsion(unsigned idx, vector<Vector>& deriv) {
  unsigned base = idx * 4;
  Vector d0 = delta(getPosition(base + 1), getPosition(base + 0));
  Vector d1 = delta(getPosition(base + 2), getPosition(base + 1));
  Vector d2 = delta(getPosition(base + 3), getPosition(base + 2));

  Vector dd0, dd1, dd2;
  PLMD::Torsion t;
  double torsion = t.compute(d0, d1, d2, dd0, dd1, dd2);

  // Convert to degrees
  double ang = torsion * 180.0 / pi;

  // Store derivatives (chain rule: d/dx = d/dang * dang/dx)
  double deg2rad = pi / 180.0;
  deriv[base + 0] = dd0 * deg2rad;
  deriv[base + 1] = (dd1 - dd0) * deg2rad;
  deriv[base + 2] = (dd2 - dd1) * deg2rad;
  deriv[base + 3] = -dd2 * deg2rad;

  return ang;
}

void AngleSwitch::buildReflist() {
  reBuildReflist = false;
  angList.clear();
  vector<Vector> dummy_deriv(getNumberOfAtoms());

  for (unsigned i = 0; i < atoms.size(); i++) {
    double ang = computeTorsion(i, dummy_deriv);
    // Snap to nearest multiple of ang_period, offset by ang_offset
    double ref = round((ang - ang_offset) / ang_period) * ang_period + ang_offset;
    if (ref > 180) ref -= 360;
    if (ref < -180) ref += 360;
    angList.push_back(ref);
  }
}

double AngleSwitch::switching_function(double ang, double ang_ref, double ang_w, double p,
                                        double prefactor, unsigned idx, vector<Vector>& deriv) {
  double angdist = ang - ang_ref;
  double sign = 1.0;

  // Handle periodicity
  if (angdist > 180) angdist -= 360;
  if (angdist < -180) angdist += 360;
  if (angdist < 0) sign = -1.0;
  angdist *= sign;

  if (angdist > ang_w) return 1.0;

  double xi = angdist / ang_w;

  // Compute gradient contribution
  if (prefactor != 0.0) {
    double dFdang = prefactor * sign * pow(xi, p - 1.0) / ang_w;
    unsigned base = idx * 4;
    for (unsigned j = 0; j < 4; j++) {
      deriv[base + j] *= dFdang;
    }
  }

  return xi;
}

void AngleSwitch::calculate() {
  if (pbc) makeWhole();

  // Build reference list if needed
  if (reBuildReflist) buildReflist();

  vector<Vector> deriv(getNumberOfAtoms());
  vector<Vector> ang_deriv(getNumberOfAtoms());
  double pairsum = 0.0;

  // First pass: compute all torsions and their switching values
  for (unsigned i = 0; i < atoms.size(); i++) {
    double ang = computeTorsion(i, ang_deriv);
    double xi = switching_function(ang, angList[i], ang_width, power, 0.0, i, ang_deriv);
    pairsum += pow(xi, power);
  }

  // Compute progress with cosine smoothing
  double progress = pow(pairsum, 1.0 / power);
  double value;
  double prefact = 0.0;

  if (progress < 1.0 && progress > 1e-10) {
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
    for (unsigned i = 0; i < atoms.size(); i++) {
      double ang = computeTorsion(i, ang_deriv);
      switching_function(ang, angList[i], ang_width, power, prefact, i, ang_deriv);
      unsigned base = i * 4;
      for (unsigned j = 0; j < 4; j++) {
        deriv[base + j] += ang_deriv[base + j];
      }
    }
  }

  // Set derivatives
  for (unsigned i = 0; i < getNumberOfAtoms(); i++) {
    setAtomsDerivatives(i, deriv[i]);
  }

  setBoxDerivativesNoPbc();
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

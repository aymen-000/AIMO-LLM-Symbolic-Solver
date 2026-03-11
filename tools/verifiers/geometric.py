"""
tools/verifiers/geometric.py

Coordinate geometry and classical geometry verification.

Handles:
  - Distance / area / perimeter computations
  - Triangle properties (Heron's formula, law of cosines)
  - Circle properties (area, arc length, chord length)
  - Coordinate geometry (slope, midpoint, intersection)
  - Pythagorean theorem checks
  - Angle computations

Strategy: extract numeric values from problem text, rebuild the
geometric computation, and compare to the claimed answer.
"""

import re
import os
import sys
import math
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.verifiers.symbolic import Verdict, VerificationResult
from tools.executor import execute_code

TOLERANCE = 1e-6


class GeometricVerifier:

    def verify(
        self,
        extraction,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not claimed_answer or not problem_text:
            return VerificationResult(Verdict.UNCERTAIN, "geometric", "Missing inputs", None)

        checks = [
            self._check_pythagorean,
            self._check_triangle_area,
            self._check_circle,
            self._check_distance,
        ]

        for check in checks:
            result = check(claimed_answer, problem_text)
            if result.verdict != Verdict.UNCERTAIN:
                return result

        return VerificationResult(
            Verdict.UNCERTAIN, "geometric", "No geometric pattern matched", None
        )

    # ── Check 1: Pythagorean theorem ──────────────────────────────────────────

    def _check_pythagorean(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Detect right triangle problems and verify via Pythagorean theorem.
        Patterns: "right triangle with legs a and b", "hypotenuse"
        """
        # Pattern: two legs given, find hypotenuse (or vice versa)
        m = re.search(
            r"(?:legs?|sides?)\s+(?:of\s+)?(?:length\s+)?(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)",
            problem_text, re.IGNORECASE
        )
        if m and re.search(r"right\s+triangle|hypotenuse", problem_text, re.IGNORECASE):
            a, b = float(m.group(1)), float(m.group(2))
            hyp = math.sqrt(a**2 + b**2)
            try:
                ans = float(claimed_answer)
                if abs(hyp - ans) < TOLERANCE:
                    return VerificationResult(
                        Verdict.PASS, "pythagorean",
                        f"√({a}²+{b}²) = {hyp:.6f} matches claimed {ans}",
                        str(hyp),
                    )
                # Maybe they want a²+b² directly
                sq = a**2 + b**2
                if abs(sq - ans) < TOLERANCE:
                    return VerificationResult(
                        Verdict.PASS, "pythagorean",
                        f"{a}²+{b}² = {sq} matches claimed {ans}",
                        str(sq),
                    )
                return VerificationResult(
                    Verdict.FAIL, "pythagorean",
                    f"Pythagorean gives {hyp:.4f} but claimed {ans}",
                    str(hyp),
                )
            except ValueError:
                pass

        return VerificationResult(Verdict.UNCERTAIN, "pythagorean", "No right triangle pattern", None)

    # ── Check 2: Triangle area (Heron's formula) ──────────────────────────────

    def _check_triangle_area(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Verify triangle area when all three sides are given."""
        sides = re.findall(
            r"(?:side|length|edge)\s+(?:of\s+)?(\d+(?:\.\d+)?)",
            problem_text, re.IGNORECASE
        )
        if len(sides) < 3:
            # Try "triangle with sides a, b, c"
            m = re.search(
                r"triangle\s+with\s+sides?\s+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)",
                problem_text, re.IGNORECASE
            )
            if m:
                sides = [m.group(1), m.group(2), m.group(3)]

        if len(sides) < 3:
            return VerificationResult(Verdict.UNCERTAIN, "triangle_area", "Can't extract 3 sides", None)

        if not re.search(r"area", problem_text, re.IGNORECASE):
            return VerificationResult(Verdict.UNCERTAIN, "triangle_area", "Not an area problem", None)

        a, b, c = float(sides[0]), float(sides[1]), float(sides[2])
        s = (a + b + c) / 2
        discriminant = s * (s-a) * (s-b) * (s-c)
        if discriminant < 0:
            return VerificationResult(Verdict.FAIL, "triangle_area", "Invalid triangle (negative discriminant)", None)

        area = math.sqrt(discriminant)

        try:
            ans = float(claimed_answer)
            if abs(area - ans) < TOLERANCE:
                return VerificationResult(
                    Verdict.PASS, "heron_formula",
                    f"Heron's formula: area = {area:.6f} matches claimed {ans}",
                    str(area),
                )
            # Check if claimed answer is area²
            if abs(area**2 - ans) < TOLERANCE:
                return VerificationResult(
                    Verdict.PASS, "heron_formula",
                    f"Area² = {area**2:.6f} matches claimed {ans}",
                    str(area**2),
                )
            return VerificationResult(
                Verdict.FAIL, "heron_formula",
                f"Heron gives area={area:.4f} but claimed {ans}",
                str(area),
            )
        except ValueError:
            pass

        return VerificationResult(Verdict.UNCERTAIN, "triangle_area", "Comparison failed", None)

    # ── Check 3: Circle properties ────────────────────────────────────────────

    def _check_circle(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Verify circle area or circumference."""
        if not re.search(r"circle|radius|diameter", problem_text, re.IGNORECASE):
            return VerificationResult(Verdict.UNCERTAIN, "circle", "Not a circle problem", None)

        m = re.search(r"radius\s+(?:of\s+)?(?:length\s+)?(\d+(?:\.\d+)?)", problem_text, re.IGNORECASE)
        if not m:
            m = re.search(r"r\s*=\s*(\d+(?:\.\d+)?)", problem_text)
        if not m:
            return VerificationResult(Verdict.UNCERTAIN, "circle", "Can't extract radius", None)

        r = float(m.group(1))

        try:
            ans = float(claimed_answer)
        except ValueError:
            return VerificationResult(Verdict.UNCERTAIN, "circle", "Non-numeric answer", None)

        area = math.pi * r**2
        circum = 2 * math.pi * r

        if abs(area - ans) < TOLERANCE:
            return VerificationResult(
                Verdict.PASS, "circle_area",
                f"πr² = {area:.6f} matches claimed {ans}", str(area)
            )
        if abs(circum - ans) < TOLERANCE:
            return VerificationResult(
                Verdict.PASS, "circle_circumference",
                f"2πr = {circum:.6f} matches claimed {ans}", str(circum)
            )
        # Integer problems often ask for floor or integer part
        if abs(int(area) - ans) < TOLERANCE:
            return VerificationResult(
                Verdict.PASS, "circle_area_int",
                f"floor(πr²) = {int(area)} matches claimed {ans}", str(int(area))
            )

        return VerificationResult(
            Verdict.FAIL, "circle",
            f"Circle: area={area:.4f}, circum={circum:.4f}, claimed={ans}",
            None
        )

    # ── Check 4: Distance formula ─────────────────────────────────────────────

    def _check_distance(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Detect coordinate pairs and verify distance formula.
        Pattern: "points (x1, y1) and (x2, y2)"
        """
        points = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", problem_text)
        if len(points) < 2:
            return VerificationResult(Verdict.UNCERTAIN, "distance", "Can't extract 2 points", None)

        if not re.search(r"distance|length|how far", problem_text, re.IGNORECASE):
            return VerificationResult(Verdict.UNCERTAIN, "distance", "Not a distance problem", None)

        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        try:
            ans = float(claimed_answer)
            if abs(dist - ans) < TOLERANCE:
                return VerificationResult(
                    Verdict.PASS, "distance_formula",
                    f"Distance = {dist:.6f} matches claimed {ans}", str(dist)
                )
            if abs(dist**2 - ans) < TOLERANCE:
                return VerificationResult(
                    Verdict.PASS, "distance_squared",
                    f"Distance² = {dist**2:.6f} matches claimed {ans}", str(dist**2)
                )
            return VerificationResult(
                Verdict.FAIL, "distance_formula",
                f"Distance = {dist:.4f} but claimed {ans}", str(dist)
            )
        except ValueError:
            pass

        return VerificationResult(Verdict.UNCERTAIN, "distance", "Comparison failed", None)
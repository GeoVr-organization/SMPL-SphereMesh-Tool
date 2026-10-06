"""How far each sphere sticks out through the SMPL surface, before and after
the fit, and on which side of the body.

Needs output/fit_state.pkl from reproduce_fit.py. Every sphere is sampled at
fibonacciPointCount points and each point gets its signed distance to the
surface (vtkImplicitPolyDataDistance via PyVista, positive outside). A
sphere's protrusion is the largest of those distances.

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/protrusion_by_sphere.py
"""

import pickle
from pathlib import Path

import numpy as np
import pyvista as pv

outputDirectory = Path(__file__).resolve().parent / "output"
with open(outputDirectory / "fit_state.pkl", "rb") as stateFile:
    state = pickle.load(stateFile)

meshVertices, meshTriangles = state["meshVertices"], state["meshTriangles"]
surface = pv.PolyData(meshVertices, np.hstack([np.full((len(meshTriangles), 1), 3), meshTriangles]).ravel())

# Fibonacci sphere: near-uniform unit directions.
fibonacciPointCount = 600
goldenAngle = np.pi * (3.0 - np.sqrt(5.0))
heights = 1.0 - 2.0 * (np.arange(fibonacciPointCount) + 0.5) / fibonacciPointCount
ringRadii = np.sqrt(1.0 - heights * heights)
azimuths = goldenAngle * np.arange(fibonacciPointCount)
unitDirections = np.column_stack((ringRadii * np.cos(azimuths), ringRadii * np.sin(azimuths), heights))


def signedDistanceToSurface(points):
    """Signed distance of each point to the SMPL surface, positive outside."""
    return np.asarray(pv.PolyData(points).compute_implicit_distance(surface)["implicit_distance"])


# Sign check: the torso centroid must come out inside (negative).
torsoProbe = meshVertices[np.abs(meshVertices[:, 0]) < 0.1].mean(axis=0)
probeDistance = signedDistanceToSurface(torsoProbe[None, :])[0]
signFlip = 1.0 if probeDistance < 0 else -1.0
print(f"torso probe {np.round(torsoProbe, 3)} distance {probeDistance:+.3f}"
      + ("" if signFlip > 0 else "  -> sign flipped so outside is positive"))

# Which way is the back? In star pose the toes point forward, so the lowest
# vertices reach further toward the front than the heel/ankle column above.
lowestHeight = meshVertices[:, 2].min()
footVertices = meshVertices[meshVertices[:, 2] < lowestHeight + 0.03]
ankleVertices = meshVertices[(meshVertices[:, 2] > lowestHeight + 0.08) & (meshVertices[:, 2] < lowestHeight + 0.12)]
toeDirection = np.sign(np.median(footVertices[:, 1]) - np.median(ankleVertices[:, 1]))
print(f"toes point toward {'+' if toeDirection > 0 else '-'}y, so the BACK is "
      f"{'-' if toeDirection > 0 else '+'}y\n")


def protrusionTable(centres, radii):
    """Per sphere: (protrusion, direction of the worst point)."""
    protrusions, worstDirections = [], []
    for centre, radius in zip(centres, radii):
        samplePoints = centre + radius * unitDirections
        distances = signFlip * signedDistanceToSurface(samplePoints)
        worstSlot = int(np.argmax(distances))
        protrusions.append(distances[worstSlot])
        worstDirections.append(unitDirections[worstSlot])
    return np.array(protrusions), np.array(worstDirections)


originalProtrusions, _ = protrusionTable(state["originalCentres"], state["originalRadii"])
fittedProtrusions, fittedDirections = protrusionTable(state["fittedCentres"], state["fittedRadii"])

print("Spheres sticking out furthest after the fit (metres outside the surface):")
print(f"{'s':>4} {'before':>8} {'after':>8} {'r0':>7} {'r_fit':>7} {'|dc|':>7}  "
      f"{'centre0':>26}  {'centre shift':>24}  worst point toward")
for sphereIndex in np.argsort(fittedProtrusions)[::-1][:10]:
    centreShift = state["fittedCentres"][sphereIndex] - state["originalCentres"][sphereIndex]
    direction = fittedDirections[sphereIndex]
    sideName = "back" if np.sign(direction[1]) == -toeDirection and abs(direction[1]) > 0.5 else (
        "front" if abs(direction[1]) > 0.5 else "side/up/down")
    print(f"{sphereIndex:>4} {originalProtrusions[sphereIndex]:8.4f} {fittedProtrusions[sphereIndex]:8.4f} "
          f"{state['originalRadii'][sphereIndex]:7.4f} {state['fittedRadii'][sphereIndex]:7.4f} "
          f"{np.linalg.norm(centreShift):7.4f}  {str(np.round(state['originalCentres'][sphereIndex], 3)):>26}  "
          f"{str(np.round(centreShift, 3)):>24}  {np.round(direction, 2)} ({sideName})")

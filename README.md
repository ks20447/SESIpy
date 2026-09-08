# SWARM-ENABLED SPATIAL INTELLIGENCE (SESI)

![SESIpy Logo](assets/images/SESIpy_logo_clear.png)

Welcome to SESIpy, an advanced 3D modelling engine for autonomous mapping and monitoring of the radio frequency domain using robotic agents. SESIpy offers 2D and 3D simulations environments as well as real-robotics implementations using ROS2. Electromagnetic propagation modelling is handled by the open-source python library LyceanEM [(github)](https://github.com/LyceanEM/LyceanEM-Python/tree/master). SESIpy incorporates these solvers into a versatile, user-friendly API for use with robotics and robotics-simulation.  

## Modules

SESIpy is broken down into a series of modules and sub-modules that concentrate on different aspects of the technology.

### Engines

#### *Spatial Intelligence*

This is the central node of the library. The spatial intelligence module is responsible for the creation, handling and simulation of electromagnetic simulations through meshes. Here, 3D scenes are initialized that contain the physical structure of the environment, as well as the desired antennas for receiving and transmitting signals. Furthermore, angle of arrival estimation and analysis is included.

#### *Mapping*

The mapping module is responsible for incorporating 2D and 3D sensor data into an agents intelligence arsenal. Included are methods to extract meta data from pgm maps and LiDAR scans in order to reconstruct a given environment. From here, agents are able to gain an understanding of the world they are operating in and can make informed, spatially-aware decisions about their sampling methods.

#### *Evaluation*

The evaluation module is concerned with the comparison of simulations against true measurements. Mainly, this will be used to improve localization accuracy by reducing the ambiguities of angle of arrival estimations, in particular NLOS transmission, towards scatterers.

## Maintaining public exports

Package `__init__.py` files are generated. Edit the declarations below, then run
`python scripts/generate_inits.py` from the repository root. The generator uses
only the Python standard library and reads source without importing scientific
libraries. Generated imports remain lazy and load only the requested module.

1. Add a literal `__all__` list in each implementation module. Include the
   functions/classes/constants you want its package to expose, for example:

   ```python
   __all__ = ["ExistingClass", "new_function"]
   ```

   Use `__all__ = []` for an internal module. Private helpers and imported names
   are not automatically exported. Declarations must be literal lists/tuples
   of names defined or explicitly imported at module level; dynamic lists and
   conditional definitions are not supported.

2. Run `python scripts/generate_inits.py`. New `.py` modules and package folders
   under `sesipy` are discovered automatically. Each package combines its direct
   modules' exports and its child packages' exports. New folders receive an
   initializer. Duplicate exports and names that clash with submodules are
   rejected rather than silently overwritten.

3. To expose a name through `from sesipy import ...`, add it to the `sesipy`
   selection in `api_exports.json`. `sesipy.simulation` also has a selection to
   preserve its smaller public API; add new names there first when promoting
   simulation helpers. Packages absent from this configuration automatically
   expose all their available exports. These selection lists intentionally
   keep the top-level API curated. Renamed/deleted selected names cause an error
   until the selection is updated.

4. Commit both the declarations and generated initializers. Do not edit generated
   initializers directly; their package docstrings are preserved. The generator
   updates lazy mappings, `__all__`, and `TYPE_CHECKING` imports together.

Check without changing files:

```sh
python scripts/generate_inits.py --check
python -B -m unittest discover -s tests -p test_export_generator.py -v
```

`--check` exits with status 1 for stale files and status 2 for invalid declarations.
The GitHub Actions workflow runs both checks without installing Sesipy's scientific
dependencies. It will run once these files are committed and pushed.

For automatic updates when committing, enable the included pre-commit configuration
once in your development environment:

```sh
python -m pip install pre-commit
pre-commit install
```

The hook regenerates initializers. If files change, review and stage them, then retry
the commit. The hook is optional; CI still catches stale generated files.

Run import regression checks in the full `lyceanem-test` environment:

```sh
python -B -m unittest discover -s tests -p test_lazy_imports.py -v
```

Reference Manual
================

This reference manual details modules, functions, and variables included in
Manim, describing what they are and what they do.  For learning how to use
Manim, see :doc:`tutorials/index`.  For a list of changes since the last release, see
the :doc:`changelog`.

.. warning:: The pages linked to here are currently a work in progress.

Inheritance Graphs
------------------

Animations
**********

.. inheritance-diagram::
   arachne.animation.animation
   arachne.animation.changing
   arachne.animation.composition
   arachne.animation.creation
   arachne.animation.fading
   arachne.animation.growing
   arachne.animation.indication
   arachne.animation.movement
   arachne.animation.numbers
   arachne.animation.rotation
   arachne.animation.specialized
   arachne.animation.speedmodifier
   arachne.animation.transform
   arachne.animation.transform_matching_parts
   arachne.animation.updaters.mobject_update_utils
   arachne.animation.updaters.update
   :parts: 1
   :top-classes: arachne.animation.animation.Animation

Cameras
*******

.. inheritance-diagram::
   arachne.camera.camera
   arachne.camera.mapping_camera
   arachne.camera.moving_camera
   arachne.camera.multi_camera
   arachne.camera.three_d_camera
   :parts: 1
   :top-classes: arachne.camera.camera.Camera, arachne.mobject.mobject.Mobject

Mobjects
********

.. inheritance-diagram::
   arachne.mobject.frame
   arachne.mobject.geometry.arc
   arachne.mobject.geometry.boolean_ops
   arachne.mobject.geometry.line
   arachne.mobject.geometry.polygram
   arachne.mobject.geometry.shape_matchers
   arachne.mobject.geometry.tips
   arachne.mobject.graph
   arachne.mobject.graphing.coordinate_systems
   arachne.mobject.graphing.functions
   arachne.mobject.graphing.number_line
   arachne.mobject.graphing.probability
   arachne.mobject.graphing.scale
   arachne.mobject.logo
   arachne.mobject.matrix
   arachne.mobject.mobject
   arachne.mobject.table
   arachne.mobject.three_d.polyhedra
   arachne.mobject.three_d.three_d_utils
   arachne.mobject.three_d.three_dimensions
   arachne.mobject.svg.brace
   arachne.mobject.svg.svg_mobject
   arachne.mobject.text.code_mobject
   arachne.mobject.text.numbers
   arachne.mobject.text.tex_mobject
   arachne.mobject.text.text_mobject
   arachne.mobject.text.typst_mobject
   arachne.mobject.types.image_mobject
   arachne.mobject.types.point_cloud_mobject
   arachne.mobject.types.vectorized_mobject
   arachne.mobject.value_tracker
   arachne.mobject.vector_field
   :parts: 1
   :top-classes: arachne.mobject.mobject.Mobject

Scenes
******

.. inheritance-diagram::
   arachne.scene.moving_camera_scene
   arachne.scene.scene
   arachne.scene.scene_file_writer
   arachne.scene.three_d_scene
   arachne.scene.vector_space_scene
   arachne.scene.zoomed_scene
   :parts: 1
   :top-classes: arachne.scene.scene.Scene, arachne.scene.scene.RerunSceneHandler


Module Index
------------

.. toctree::
   :maxdepth: 3

   reference_index/animations
   reference_index/cameras
   reference_index/configuration
   reference_index/mobjects
   reference_index/scenes
   reference_index/utilities_misc

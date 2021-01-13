final class GeometricPrimitive: Boundable, Intersectable, Material {

        init(shape: Shape, material: Material) {
                self.shape = shape
                self.material = material
        }

        func intersect(ray: Ray, tHit: inout FloatX) throws -> SurfaceInteraction? {
                var shapeInteraction = try shape.intersect(ray: ray, tHit: &tHit)
                shapeInteraction?.primitive = self
                return shapeInteraction
        }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                return material.computeScatteringFunctions(interaction: interaction)
        }

        var shape: Shape
        var material: Material
}

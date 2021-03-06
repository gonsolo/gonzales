final class AreaLight: Light, Boundable, Intersectable, Material {

        init(brightness: Spectrum, shape: Shape) {
                self.brightness = brightness
                self.shape = shape
        }

        func emittedRadiance(from interaction: Interaction, inDirection direction: Vector) -> Spectrum {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(for ref: Interaction, u: Point2F) -> (radiance: Spectrum, direction: Vector, pdf: FloatX, visibility: Visibility) {
                let (shapeInteraction, pdf) = shape.sample(ref: ref, u: u)
                let direction: Vector = normalized(shapeInteraction.position - ref.position)
                assert(!direction.isNaN)
                let visibility = Visibility(from: ref, to: shapeInteraction)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return (radiance, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction) throws -> FloatX {
                return try shape.probabilityDensityFor(samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for ray: Ray) -> Spectrum { return black }

        var isDelta: Bool { get { return false } }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func intersect(ray: Ray, tHit: inout FloatX) throws -> SurfaceInteraction? {
                var interaction = try shape.intersect(ray: ray, tHit: &tHit)
                interaction?.primitive = self
                return interaction
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                let matte = Matte(kd: ConstantTexture(value: white))
                let (bsdf, bssrdf) = matte.computeScatteringFunctions(interaction: interaction)
                return (bsdf, bssrdf)
        }

        let shape: Shape
        let brightness: Spectrum
}


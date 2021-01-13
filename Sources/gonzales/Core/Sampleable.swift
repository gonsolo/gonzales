/**
        A type that can be sampled.
        For example, to render area lights a renderer typically chooses a point
        on the area light and calculates the light from there. If many such
        points are chosen and the light calculated, the average approximates
        the analytical solution.
*/
protocol Sampleable {
        func sample(u: Point2F) -> (interaction: Interaction, pdf: FloatX)
        func sample(ref: Interaction, u: Point2F) -> (Interaction, FloatX)
        func probabilityDensityFor(samplingDirection direction: Vector, from interaction: Interaction) throws -> FloatX
        func area() -> FloatX
}

extension Sampleable {
        func sample(ref: Interaction, u: Point2F) -> (Interaction, FloatX) {
                var (intr, pdf) = sample(u: u)
                let wi: Vector = normalized(intr.position - ref.position)
                pdf *= distanceSquared(ref.position, intr.position) / absDot(Vector(normal: intr.normal), -wi)
                return (intr, pdf)
        }
}

extension Sampleable where Self: Intersectable {
        func probabilityDensityFor(samplingDirection direction: Vector, from interaction: Interaction) throws -> FloatX {
                let ray = interaction.spawnRay(inDirection: direction)
                var tHit: FloatX = 0.0
                guard let isect = try intersect(ray: ray, tHit: &tHit) else { return 0 }
                var pdf = distanceSquared(interaction.position, isect.position) / (absDot(isect.normal, -direction) * area())
                if pdf.isInfinite { pdf = 0 }
                return pdf
        }
}


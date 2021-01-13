struct SurfaceInteraction: Interaction {

        init(position: Point = Point(),
             normal: Normal = Normal(),
             shadingNormal: Normal = Normal(),
             wo: Vector = Vector(),
             dpdu: Vector = Vector(),
             uv: Point2F = Point2F(),
             faceIndex: Int = 0,
             primitive: (Boundable & Intersectable)? = nil) {

                self.position = position
                self.normal = normal
                self.shadingNormal = shadingNormal
                self.wo = wo
                self.dpdu = dpdu
                self.uv = uv
                self.faceIndex = faceIndex
                self.primitive = primitive
        }

        init(_ other: SurfaceInteraction) {
                self.position = other.position
                self.normal = other.normal
                self.shadingNormal = other.shadingNormal
                self.wo = other.wo
                self.dpdu = other.dpdu
                self.uv = other.uv
                self.faceIndex = other.faceIndex
                self.primitive = other.primitive
        }

        var position: Point
        var normal: Normal
        var shadingNormal: Normal
        var wo: Vector
        var dpdu: Vector
        var uv: Point2F
        var faceIndex: Int

        var primitive: (Boundable & Intersectable)?
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}


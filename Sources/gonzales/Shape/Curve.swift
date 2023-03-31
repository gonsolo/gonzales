import Foundation  // sqrt, log2

var numberOfCurves = 0

enum CurveError: Error {
        case index
        case numberControlPoints
        case todo
        case accelerator
}

typealias TwoPoints = (Point, Point)
typealias ThreePoints = (Point, Point, Point)
typealias FourPoints = (Point, Point, Point, Point)
typealias SevenPoints = (Point, Point, Point, Point, Point, Point, Point)

typealias TwoFloats = (FloatX, FloatX)
typealias ThreeFloats = (FloatX, FloatX, FloatX)

private func lerp(with t: FloatX, between first: Point, and second: Point) -> Point {
        return (1.0 - t) * first + t * second
}

func blossomBezier(points: FourPoints, u: ThreeFloats) -> Point {
        var a: ThreePoints
        a.0 = lerp(with: u.0, between: points.0, and: points.1)
        a.1 = lerp(with: u.0, between: points.1, and: points.2)
        a.2 = lerp(with: u.0, between: points.2, and: points.3)
        var b: TwoPoints
        b.0 = lerp(with: u.1, between: a.0, and: a.1)
        b.1 = lerp(with: u.1, between: a.1, and: a.2)
        return lerp(with: u.2, between: b.0, and: b.1)
}

func blossomBezier(points: FourPoints, u: TwoFloats) -> FourPoints {
        var p: FourPoints
        p.0 = blossomBezier(points: points, u: (u.0, u.0, u.0))
        p.1 = blossomBezier(points: points, u: (u.0, u.0, u.1))
        p.2 = blossomBezier(points: points, u: (u.0, u.1, u.1))
        p.3 = blossomBezier(points: points, u: (u.1, u.1, u.1))
        return p
}

extension Point2 {
        init(_ point: Point3<T>) {
                self.x = point.x
                self.y = point.y
        }
}

extension Vector2 {
        init(_ point: Point3<T>) {
                self.x = point.x
                self.y = point.y
        }
}

final class Curve: Shape {

        init(objectToWorld: Transform, common: CurveCommon, u: TwoFloats) {
                self.common = common
                self.u = u
                self.objectToWorld = objectToWorld
                numberOfCurves += 1
        }

        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        func objectBound() -> Bounds3f {
                let points = blossomBezier(points: common.points, u: u)
                let bounds = union(
                        first: Bounds3f(first: points.0, second: points.1),
                        second: Bounds3f(first: points.2, second: points.3))
                let maxWidth = computeMaxWidth(u: u, width: common.width)
                let expanded = expand(bounds: bounds, by: maxWidth * 0.5)
                return expanded
        }

        private func subdivideBezier(points: [Point]) -> [Point] {
                var splitPoints = [Point]()
                splitPoints.append(points[0])
                splitPoints.append((points[0] + points[1]) / 2)
                splitPoints.append((points[0] + 2 * points[1] + points[2]) / 4)
                splitPoints.append(
                        (points[0] + Point(3 * points[1]) + Point(3 * points[2]) + points[3]) / 8)
                splitPoints.append((points[1] + 2 * points[2] + points[3]) / 4)
                splitPoints.append((points[2] + points[3]) / 2)
                splitPoints.append(points[3])
                return splitPoints
        }

        func evalBezier(points: [Point], u: FloatX) -> (Point, Vector) {
                let p = (points[0], points[1], points[2], points[3])
                return evalBezier(points: p, u: u)
        }

        func evalBezier(points: FourPoints, u: FloatX) -> (Point, Vector) {
                let cp1 = ThreePoints(
                        lerp(with: u, between: points.0, and: points.1),
                        lerp(with: u, between: points.1, and: points.2),
                        lerp(with: u, between: points.2, and: points.3))
                let cp2 = TwoPoints(
                        lerp(with: u, between: cp1.0, and: cp1.1),
                        lerp(with: u, between: cp1.1, and: cp1.2))
                var derivative: Vector
                let d: Vector = cp2.1 - cp2.0
                if lengthSquared(d) > 0 {
                        derivative = 3 * d
                } else {
                        derivative = points.3 - points.0
                }
                let point = lerp(with: u, between: cp2.0, and: cp2.1)
                return (point, derivative)
        }

        func recursiveIntersect(
                ray: Ray,
                tHit: inout FloatX,
                points: [Point],
                index: Int = 0,
                rayToObject: Transform,
                u: TwoFloats,
                depth: Int
        ) throws -> (SurfaceInteraction, FloatX) {

                let nothing: (SurfaceInteraction, FloatX) = (SurfaceInteraction(), FloatX.infinity)

                let rayLength = length(ray.direction)
                if depth > 0 {
                        let splitPoints = subdivideBezier(points: points)
                        let u = [u.0, (u.0 + u.1) / 2, u.1]
                        typealias InteractionAndT = (SurfaceInteraction, FloatX)
                        var hits: [InteractionAndT] = [
                                (SurfaceInteraction(), 0),
                                (SurfaceInteraction(), 0),
                        ]

                        for segment in 0..<2 {
                                let cps = segment * 3
                                let maxWidth = computeMaxWidth(
                                        u: (u[segment], u[segment + 1]), width: common.width)
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 1, width: maxWidth)
                                {
                                        continue
                                }
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 0, width: maxWidth)
                                {
                                        continue
                                }
                                let zMax = rayLength * tHit
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 2, width: maxWidth,
                                        limits: (0, zMax))
                                {
                                        continue
                                }
                                hits[segment] = try recursiveIntersect(
                                        ray: ray,
                                        tHit: &tHit,
                                        points: splitPoints,
                                        index: cps,
                                        rayToObject: rayToObject,
                                        u: (u[segment], u[segment + 1]),
                                        depth: depth - 1)
                                // if hit && !tHit: shadowRays not applicable here
                        }
                        if hits[0].0.valid && hits[1].0.valid {
                                if hits[0].1 < hits[1].1 {
                                        return hits[0]
                                } else {
                                        return hits[1]
                                }
                        } else if hits[0].0.valid && !hits[1].0.valid {
                                return hits[0]
                        } else if !hits[0].0.valid && hits[1].0.valid {
                                return hits[1]
                        } else {
                                return nothing
                        }
                } else {
                        func testTangent(_ a: Int, _ b: Int) -> Bool {
                                let edge =
                                        (points[a].y - points[b].y) * -points[b].y + points[b].x
                                        * (points[b].x - points[a].x)
                                if edge < 0 {
                                        return false
                                } else {
                                        return true
                                }
                        }

                        if !testTangent(1, 0) { return nothing }
                        if !testTangent(2, 3) { return nothing }
                        let segmentDirection: Vector2 = Point2F(points[3]) - Point2F(points[0])
                        let denominator = lengthSquared(segmentDirection)
                        if denominator == 0 { return nothing }
                        let w = dot(-Vector2F(points[0]), segmentDirection) / denominator
                        let nu = clamp(
                                value: lerp(with: w, between: u.0, and: u.1), low: u.0, high: u.1)
                        let hitWidth = lerp(with: nu, between: common.width.0, and: common.width.1)
                        let (pc, dpcdw) = evalBezier(
                                points: points, u: clamp(value: w, low: 0, high: 1))
                        let ptCurveDist2 = pc.x * pc.x + pc.y * pc.y
                        if ptCurveDist2 > hitWidth * hitWidth * 0.25 { return nothing }
                        let zMax = rayLength * tHit
                        if pc.z < 0 || pc.z > zMax { return nothing }
                        let ptCurveDist = sqrt(ptCurveDist2)
                        let edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y
                        var v: FloatX
                        if edgeFunc > 0 {
                                v = 0.5 + ptCurveDist / hitWidth
                        } else {
                                v = 0.5 - ptCurveDist / hitWidth
                        }
                        // if tHit != nullptr
                        let tHit = pc.z / rayLength
                        //let pError = Vector(x: 2 * hitWidth, y: 2 * hitWidth, z: 2 * hitWidth)
                        let (_, dpdu) = evalBezier(points: common.points, u: nu)
                        let dpduPlane = rayToObject.inverse * dpdu
                        let dpdvPlane =
                                normalized(Vector(x: -dpduPlane.y, y: dpduPlane.x, z: 0)) * hitWidth
                        let dpdv = rayToObject * dpdvPlane
                        let normal = Normal(normalized(cross(dpdu, dpdv)))
                        let uvHit = Point2F(x: nu, y: v)
                        let pHit = ray.getPointFor(parameter: tHit)
                        let localInteraction = SurfaceInteraction(
                                position: pHit,
                                normal: normal,
                                shadingNormal: normal,
                                wo: -ray.direction,
                                dpdu: dpdu,
                                uv: uvHit,
                                faceIndex: 0)
                        let worldInteraction = objectToWorld * localInteraction
                        let validWorldInteraction = SurfaceInteraction(
                                valid: true,
                                position: worldInteraction.position,
                                normal: worldInteraction.normal,
                                shadingNormal: worldInteraction.shadingNormal,
                                wo: worldInteraction.wo,
                                dpdu: worldInteraction.dpdu,
                                uv: worldInteraction.uv,
                                faceIndex: worldInteraction.faceIndex,
                                material: worldInteraction.material)
                        return (validWorldInteraction, tHit)
                }
        }

        private func computeMaxWidth(u: TwoFloats, width: TwoFloats) -> FloatX {
                let width = computeWidth(u: u, width: width)
                let result = max(width.0, width.1)
                return result
        }

        private func computeWidth(u: TwoFloats, width: TwoFloats) -> (FloatX, FloatX) {
                let result = (
                        lerp(with: u.0, between: width.0, and: width.1),
                        lerp(with: u.1, between: width.0, and: width.1)
                )
                return result
        }

        private func overlap(
                points: [Point], index: Int = 0, xyz: Int, width: FloatX, limits: TwoFloats = (0, 0)
        ) -> Bool {
                let pMax = max(
                        max(points[index + 0][xyz], points[index + 1][xyz]),
                        max(points[index + 2][xyz], points[index + 3][xyz]))
                let pMin = min(
                        min(points[index + 0][xyz], points[index + 1][xyz]),
                        min(points[index + 2][xyz], points[index + 3][xyz]))
                var overlaps: Bool
                if pMax + 0.5 * width < limits.0 || pMin - 0.5 * width > limits.1 {
                        overlaps = false
                } else {
                        overlaps = true
                }
                return overlaps
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                let ray = worldToObject * worldRay
                let points = blossomBezier(points: common.points, u: u)
                let dx = cross(ray.direction, points.3 - points.0)
                if lengthSquared(dx) == 0 {
                        throw CurveError.todo
                }
                let objectToRay = try lookAtTransform(
                        eye: ray.origin, at: ray.origin + ray.direction, up: dx)
                var controlPoints = [Point(), Point(), Point(), Point()]
                controlPoints[0] = objectToRay * points.0
                controlPoints[1] = objectToRay * points.1
                controlPoints[2] = objectToRay * points.2
                controlPoints[3] = objectToRay * points.3
                let maxWidth = computeMaxWidth(u: u, width: common.width)
                if !overlap(points: controlPoints, xyz: 1, width: maxWidth) {
                        return
                }
                if !overlap(points: controlPoints, xyz: 0, width: maxWidth) {
                        return
                }
                let rayLength = length(ray.direction)
                let zMax = rayLength * tHit
                if !overlap(points: controlPoints, xyz: 2, width: maxWidth, limits: (0, zMax)) {
                        return
                }
                var l0: FloatX = 0
                for i in 0..<2 {
                        let px: FloatX = abs(
                                controlPoints[i].x - 2.0 * controlPoints[i + 1].x
                                        + controlPoints[i + 2].x)
                        let py: FloatX = abs(
                                controlPoints[i].y - 2.0 * controlPoints[i + 1].y
                                        + controlPoints[i + 2].y)
                        let pz: FloatX = abs(
                                controlPoints[i].z - 2.0 * controlPoints[i + 1].z
                                        + controlPoints[i + 2].z)
                        l0 = max(max(l0, px), max(py, pz))
                }
                let eps = max(common.width.0, common.width.1) * 0.05
                let r0 = Int(log2(1.41421356237 * 6.0 * l0 / (8.0 * eps)) / 2.0)
                let maxDepth = clamp(value: r0, low: 0, high: 10)
                let interactionAndT = try recursiveIntersect(
                        ray: ray,
                        tHit: &tHit,
                        points: controlPoints,
                        rayToObject: objectToRay.inverse,
                        u: u,
                        depth: maxDepth)
                tHit = interactionAndT.1
                interaction = interactionAndT.0
                interaction.material = material
        }

        func area() -> FloatX {
                fatalError("Not implemented")
        }

        func sample(u: TwoFloats) -> (interaction: Interaction, pdf: FloatX) {
                fatalError("Not implemented")
        }

        public var description: String {
                return "Curve"
        }

        static func statistics() {
                print("  Number of curves:\t\t\t\t\t\t\t\(numberOfCurves)")
        }

        var common: CurveCommon
        var u: TwoFloats
        var objectToWorld: Transform
}

struct CurveCommon {

        let points: FourPoints
        let width: TwoFloats
}

func createCurve(objectToWorld: Transform, points: FourPoints, width: TwoFloats) -> [Shape] {
        let common = CurveCommon(points: points, width: width)
        // The default splitdepth in pbrt is 3 which would make this 8 segments (1 << splitDepth).
        // Let's use 4 for the time being to save a little bit of memory.
        let numberOfSegments = 4
        var segments = [Shape]()
        for i in 0..<numberOfSegments {
                let n = FloatX(numberOfSegments)
                let uMin = FloatX(i) / n
                let uMax = (FloatX(i) + 1) / n
                let curve = Curve(objectToWorld: objectToWorld, common: common, u: (uMin, uMax))
                segments.append(curve)
        }
        return segments
}

func createBVHCurveShape(
        controlPoints: [Point],
        widths: (Float, Float),
        objectToWorld: Transform,
        degree: Int
) -> [Shape] {
        var curves = [Shape]()
        let numberOfSegments = controlPoints.count - degree
        for segment in 0..<numberOfSegments {
                var bezierPoints: FourPoints
                let p012 = controlPoints[segment + 0]
                let p123 = controlPoints[segment + 1]
                let p234 = controlPoints[segment + 2]
                let p345 = controlPoints[segment + 3]
                let p122 = lerp(with: 2.0 / 3.0, between: p012, and: p123)
                let p223 = lerp(with: 1.0 / 3.0, between: p123, and: p234)
                let p233 = lerp(with: 2.0 / 3.0, between: p123, and: p234)
                let p334 = lerp(with: 1.0 / 3.0, between: p234, and: p345)
                let p222 = lerp(with: 0.5, between: p122, and: p223)
                let p333 = lerp(with: 0.5, between: p233, and: p334)
                bezierPoints.0 = p222
                bezierPoints.1 = p223
                bezierPoints.2 = p233
                bezierPoints.3 = p333
                let w0 = lerp(
                        with: FloatX(segment) / FloatX(numberOfSegments), between: widths.0,
                        and: widths.1)
                let w1 = lerp(
                        with: FloatX(segment + 1) / FloatX(numberOfSegments), between: widths.0,
                        and: widths.1)
                let curve = createCurve(
                        objectToWorld: objectToWorld,
                        points: bezierPoints,
                        width: (w0, w1))

                curves.append(contentsOf: curve)
        }
        return curves
}

func createCurveShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [Shape] {
        let degree = 3
        let controlPoints = try parameters.findPoints(name: "P")
        let width = try parameters.findOneFloatX(called: "width", else: 0.5)
        let width0 = try parameters.findOneFloatX(called: "width0", else: width)
        let width1 = try parameters.findOneFloatX(called: "width1", else: width)
        guard controlPoints.count >= degree + 1 else {
                throw CurveError.numberControlPoints
        }
        var curves = [Shape]()
        switch acceleratorName {
        case "bvh":
                curves = createBVHCurveShape(
                        controlPoints: controlPoints,
                        widths: (width0, width1),
                        objectToWorld: objectToWorld,
                        degree: degree)
        case "embree":
                curves = createEmbreeCurveShape(
                        controlPoints: controlPoints,
                        widths: (width0, width1),
                        objectToWorld: objectToWorld
                )
        default:
                throw CurveError.accelerator
        }
        return curves
}

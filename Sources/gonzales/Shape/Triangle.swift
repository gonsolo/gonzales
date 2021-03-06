var triangleIntersections = 0
var triangleHits = 0

var numberOfTriangles = 0
var triangleMemory = 0
var worldBoundCalled = 0

final class TriangleMesh<VertexIndex, Point, Normal, UVVector> {

        init(objectToWorld: Transform,
             numberTriangles: Int,
             vertexIndices: [VertexIndex],
             points: [Point],
             normals: [Normal],
             uvs: [UVVector],
             faceIndices: [VertexIndex]) {
                self.objectToWorld = objectToWorld
                self.vertexIndices = vertexIndices
                self.numberTriangles = numberTriangles
                self.points = points
                self.normals = normals
                self.uvs = uvs
                self.faceIndices = faceIndices
        }

        let objectToWorld:      Transform
        let vertexIndices:      [VertexIndex]
        let numberTriangles:     Int
        let points:             [Point]
        let normals:            [Normal]
        let uvs:                [UVVector]
        let faceIndices:        [VertexIndex]
}

enum TriangleError: Error {
        case index
        case vertexIndices
}

extension Array {

        subscript<Index: BinaryInteger>(i: Index) -> Element {
                get {
                        return self[Int(i)]
                }
        }
}

final class Triangle<VertexIndex: BinaryInteger>: Shape {

        init(mesh: TriangleMesh<VertexIndex, Point, Normal, Vector2F>,
             number: VertexIndex) throws {
                self.mesh = mesh
                self.idx = 3 * number
                numberOfTriangles += 1
                triangleMemory += MemoryLayout<Self>.stride

                guard vertexIndex0 < mesh.points.count &&
                      vertexIndex1 < mesh.points.count &&
                      vertexIndex2 < mesh.points.count else {
                        throw TriangleError.index
                }
        }

        static func formatHuman(_ number: Int) -> String {
                if number > 1024 * 1024 {
                        return String(format: "%.1f", Double(number) / 1024.0 / 1024.0) + "MB"
                }
                if number > 1024 {
                        return String(number / 1024) + "KB"
                }
                return String(number) + " bytes"
        }

        static func statistics() {
                print("  Number of triangles:\t\t\t\t\t\t\t\(numberOfTriangles)")
                print("  Triangle memory:\t\t\t\t\t\t\t\(formatHuman(triangleMemory))")
                print("  Number of triangle intersection tests:\t\t\t\t\(triangleIntersections)")
                print("  Triangle worldBound calls:\t\t\t\t\t\t\(worldBoundCalled)")
        }

        var vertexIndex0: VertexIndex { get { return mesh.vertexIndices[idx + 0] } }
        var vertexIndex1: VertexIndex { get { return mesh.vertexIndices[idx + 1] } }
        var vertexIndex2: VertexIndex { get { return mesh.vertexIndices[idx + 2] } }

        func objectBound() -> Bounds3f {
                let (p0, p1, p2) = getLocalPoints()
                return Union(bound: Bounds3f(first: p0, second: p1), point: p2)
        }

        func worldBound() -> Bounds3f {
                worldBoundCalled += 1
                return objectToWorld * objectBound()
        }

        func getUVs() -> (Vector2F, Vector2F, Vector2F) {
                if mesh.uvs.isEmpty {
                        let z = Vector2F()
                        return (z, z, z)
                } else {
                        let uvs = mesh.uvs
                        return (Vector2F(uvs[vertexIndex0]),
                                Vector2F(uvs[vertexIndex1]),
                                Vector2F(uvs[vertexIndex2]))
                }
        }

        func intersect(ray worldRay: Ray, tHit: inout FloatX) throws -> SurfaceInteraction? {

                let empty = {(line: Int) -> SurfaceInteraction? in
                        //print("No triangle intersection at line ", line)
                        //Thread.callStackSymbols.forEach { print($0) }
                        return nil
                }

                let ray = worldToObject * worldRay

                triangleIntersections += 1

                let (p0, p1, p2) = getLocalPoints()
                
                var p0t: Point = p0 - ray.origin
                var p1t: Point = p1 - ray.origin
                var p2t: Point = p2 - ray.origin

                let kz = maxDimension(abs(ray.direction))
                var kx = kz + 1
                if kx == 3 { kx = 0 }
                var ky = kx + 1
                if ky == 3 { ky = 3 }
                let d: Vector = permute(vector: ray.direction, x: kx, y: ky, z: kz)
                p0t = permute(point: p0t, x: kx, y: ky, z: kz)
                p1t = permute(point: p1t, x: kx, y: ky, z: kz)
                p2t = permute(point: p2t, x: kx, y: ky, z: kz)

                let Sx: FloatX = -d.x / d.z
                let Sy: FloatX = -d.y / d.z
                let Sz: FloatX = 1.0 / d.z
                p0t.x += Sx * p0t.z
                p0t.y += Sy * p0t.z
                p1t.x += Sx * p1t.z
                p1t.y += Sy * p1t.z
                p2t.x += Sx * p2t.z
                p2t.y += Sy * p2t.z

                let e0: FloatX = p1t.x * p2t.y - p1t.y * p2t.x
                let e1: FloatX = p2t.x * p0t.y - p2t.y * p0t.x
                let e2: FloatX = p0t.x * p1t.y - p0t.y * p1t.x

                if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) { 
                        return empty(#line)
                }
                let det: FloatX = e0 + e1 + e2
                if det == 0 { 
                        return empty(#line)
                }

                p0t.z *= Sz
                p1t.z *= Sz
                p2t.z *= Sz
                let tScaled: FloatX = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z
                if det < 0 && (tScaled >= 0 || tScaled < tHit * det) { return empty(#line) }
                else if det > 0 && (tScaled <= 0 || tScaled > tHit * det) { return empty(#line) }

                let invDet: FloatX = 1 / det
                let b0: FloatX = e0 * invDet
                let b1: FloatX = e1 * invDet
                let b2: FloatX = e2 * invDet
                let t: FloatX = tScaled * invDet

                let hit0: Point = b0 * p0
                let hit1: Point = b1 * p1
                let hit2: Point = b2 * p2
                let pHit: Point = hit0 + hit1 + hit2

                let dp02 = Vector(point: p0 - p2)
                let dp12 = Vector(point: p1 - p2)
                let normal = normalized(Normal(cross(dp02, dp12)))

                let uv = getUVs()
                let uvHit0: Point2F = b0 * Point2F(from: uv.0)
                let uvHit1: Point2F = b1 * Point2F(from: uv.1)
                let uvHit2: Point2F = b2 * Point2F(from: uv.2)
                let uvHit: Point2F = uvHit0 + uvHit1 + uvHit2

                let duv02: Vector2F = uv.0 - uv.2
                let duv12: Vector2F = uv.1 - uv.2
                let determinantUV: FloatX = duv02[0] * duv12[1] - duv02[1] * duv12[0]
                let degenerateUV: Bool = abs(determinantUV) < 1e-8
                var dpdu = up
                var dpdv = up

                if !degenerateUV {
                        let invDeterminantUV = 1 / determinantUV
			                  let a: Vector = +duv12[1] * dp02
			                  let b: Vector = +duv02[1] * dp12
			                  let c: Vector = -duv12[0] * dp02
			                  let d: Vector = +duv02[0] * dp12
			                  dpdu = (a - b) * invDeterminantUV
			                  dpdv = (c - d) * invDeterminantUV
                }

                if degenerateUV || lengthSquared(cross(dpdu, dpdv)) == 0 {
                        let ng: Vector = cross(p2 - p0, p1 - p0)
                        if lengthSquared(ng) == 0 { return nil }
                        (dpdu, dpdv) = makeCoordinateSystem(from: normalized(ng));
                }

                var shadingNormal: Normal
                if mesh.normals.isEmpty {
                        shadingNormal = normal
                } else {
                        let sn0 = b0 * Normal(mesh.normals[vertexIndex0])
                        let sn1 = b1 * Normal(mesh.normals[vertexIndex1])
                        let sn2 = b2 * Normal(mesh.normals[vertexIndex2])
                        shadingNormal = sn0 + sn1 + sn2
                        if lengthSquared(shadingNormal) > 0 {
                                shadingNormal = normalized(shadingNormal);
                        } else {
                                shadingNormal = Normal(x: 6, y: 6, z: 6)
                        }
                }

                var ss = normalized(dpdu)
                var ts = cross(ss, Vector(normal: shadingNormal))
                if lengthSquared(ts) > 0 {
                        ts.normalize()
                        ss = cross(ts, Vector(normal: shadingNormal))
                } else {
                        (ss, ts) = makeCoordinateSystem(from: Vector(normal: shadingNormal))
                }

                // Set shading geometry
                dpdu = ss

                var faceIndex: VertexIndex = 0
                if !mesh.faceIndices.isEmpty {
                        faceIndex = mesh.faceIndices[idx / 3]
                }

                let localInteraction = SurfaceInteraction(position: pHit,
                                                        normal: normal,
                                                        shadingNormal: shadingNormal,
                                                        wo: -ray.direction,
                                                        dpdu: dpdu,
                                                        uv: uvHit,
                                                        faceIndex: Int(faceIndex))
                let worldInteraction = objectToWorld * localInteraction
                tHit = t
                triangleHits += 1
                return worldInteraction
        }
        
        private func getLocalPoint(index: VertexIndex) -> Point {
                return Point(mesh.points[index])
        }

        private func getLocalPoints() -> (Point, Point, Point) {
                let p0 = getLocalPoint(index: vertexIndex0)
                let p1 = getLocalPoint(index: vertexIndex1)
                let p2 = getLocalPoint(index: vertexIndex2)
                return (p0, p1, p2)
        }
 
        private func uniformSampleTriangle(u: Point2F) -> Point2F {
                let su0 = u[0].squareRoot()
                return Point2F(x: 1 - su0, y: u[1] * su0)
        }

        func area() -> FloatX {
                let (p0, p1, p2) = getLocalPoints()
                return 0.5 * length(cross(Vector(vector: (p1 - p0)), p2 - p0))
        }

        func sample(u: Point2F) -> (interaction: Interaction, pdf: FloatX) {
                let b = uniformSampleTriangle(u: u)
                let (p0, p1, p2) = getLocalPoints()
                let sampled0: Point = b[0] * p0
                let sampled1: Point = b[1] * p1
                let sampled2: Point = (1 - b[0] - b[1]) * p2
                let localPoint: Point = sampled0 + sampled1 + sampled2
                let localNormal = normalized(Normal(cross(p1 - p0, p2 - p0)))
                let worldPoint = objectToWorld * localPoint
                let worldNormal = objectToWorld * localNormal
                let worldInteraction = SurfaceInteraction(position: worldPoint, normal: worldNormal)
                let pdf = 1 / area()
                return (worldInteraction, pdf)
        }

        var description: String {
                var d = "Triangle [ "
                let (p0, p1, p2) = getLocalPoints()
                d += p0.description + p1.description + p2.description
                d += " ]"
                return d
        }

        var objectToWorld: Transform {
                get {
                        return mesh.objectToWorld
                }
        }

        var mesh: TriangleMesh<VertexIndex, Point, Normal, Vector2F>

        // Can't be something else than Int because Array subscripting is not
        // possible/slow with UInt8 or others.
        let idx: VertexIndex
}

func createTriangleMeshShape(objectToWorld: Transform,
                             parameters: ParameterDictionary) throws -> [Shape] {
        let indices = try parameters.findInts(name: "indices")
        guard indices.count % 3 == 0 else {
                throw ApiError.input(message: "Triangle indices must be multiplies of 3")
        }
        let points = try parameters.findPoints(name: "P")
        let normals = try parameters.findNormals(name: "N")
        let uvFloatXs = try parameters.findFloatXs(called: "uv")
        var uvs = [Vector2F]()
        for i in 0 ..< uvFloatXs.count / 2 {
                uvs.append(Vector2F(x: uvFloatXs[2*i], y: uvFloatXs[2*i+1]))
        }
        let faceIndices = try parameters.findInts(name: "faceIndices")

        if indices.count <= UInt8.max && faceIndices.count <= UInt8.max {
                let indices8 = indices.map { UInt8($0)}
                let faceIndices8 = faceIndices.map { UInt8($0)}
                return try createTriangleMesh(objectToWorld: objectToWorld,
                                              indices: indices8,
                                              points: points,
                                              normals: normals,
                                              uvs: uvs,
                                              faceIndices: faceIndices8)

        }
        if indices.count <= UInt16.max && faceIndices.count <= UInt16.max {
                let indices16 = indices.map { UInt16($0)}
                let faceIndices16 = faceIndices.map { UInt16($0)}
                return try createTriangleMesh(objectToWorld: objectToWorld,
                                              indices: indices16,
                                              points: points,
                                              normals: normals,
                                              uvs: uvs,
                                              faceIndices: faceIndices16)
        }
        if indices.count <= UInt32.max && faceIndices.count <= UInt32.max {
                let indices32 = indices.map { UInt32($0)}
                let faceIndices32 = faceIndices.map { UInt32($0)}
                return try createTriangleMesh(objectToWorld: objectToWorld,
                                              indices: indices32,
                                              points: points,
                                              normals: normals,
                                              uvs: uvs,
                                              faceIndices: faceIndices32)
        }

        return try createTriangleMesh(objectToWorld: objectToWorld,
                                      indices: indices,
                                      points: points,
                                      normals: normals,
                                      uvs: uvs,
                                      faceIndices: faceIndices)
}

func createTriangleMesh<Index: BinaryInteger>(objectToWorld: Transform,
                        indices: [Index],
                        points: [Point],
                        normals: [Normal],
                        uvs: [Vector2F],
                        faceIndices: [Index]) throws -> [Shape] {
        let numberTriangles = indices.count / 3
        let trianglePoints = points
        let triangleNormals  = normals
        let triangleUvs = uvs
        var triangles = [Triangle<Index>]()

        let mesh = TriangleMesh<Index, Point, Normal, Vector2F>(
                objectToWorld: objectToWorld,
                numberTriangles: numberTriangles,
                vertexIndices: indices,
                points: trianglePoints,
                normals: triangleNormals,
                uvs: triangleUvs,
                faceIndices: faceIndices)
        for i in 0..<numberTriangles {
                triangles.append(try Triangle<Index>(mesh: mesh, number: Index(i)))
        }
        return triangles
}

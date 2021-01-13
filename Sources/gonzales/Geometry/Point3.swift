public struct Point3<T> {

        init(x: T, y: T, z: T) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(_ point: Point3) {
                self.x = point.x
                self.y = point.y
                self.z = point.z
        }

        init(xyz: (T, T, T)) {
                self.x = xyz.0
                self.y = xyz.1
                self.z = xyz.2
        }

        subscript(index: Int) -> T {

                get {
                        switch index {
                        case 0: return x
                        case 1: return y
                        case 2: return z
                        default: return x
                        }
                }

                set(newValue) {
                        switch index {
                        case 0: x = newValue
                        case 1: y = newValue
                        case 2: z = newValue
                        default: break
                        }
                }
        }

        public var x: T
        public var y: T
        public var z: T
}

extension Point3: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) \(z) ]"
        }
}

func permute<T>(point: Point3<T>, x: Int, y: Int, z: Int) -> Point3<T> {
        return Point3(x: point[x], y: point[y], z: point[z])
}

public typealias Point = Point3<FloatX>

extension Point3: Three where T: FloatingPoint {
        init() {
                self.init(x: 0, y: 0, z: 0)
        }
}

extension Point3 where T: FloatingPoint {
        init(_ normal: Normal3<T>) {
                self.init(x: normal.x,
                          y: normal.y,
                          z: normal.z)
        }
}

let origin = Point()

extension Point3 where T: FloatingPoint {

        public static func * (mul: T, point: Point3<T>) -> Point3 {
                return Point3(x: point.x * mul, y: point.y * mul, z: point.z * mul)
        }

        public static func / (point: Point3<T>, divisor: T) -> Point3 {
                return Point3(x: point.x / divisor, y: point.y / divisor, z: point.z / divisor)
        }

        public static func + (left: Point3<T>, right: Point3<T>) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func + (left: Point3<T>, right: Vector3<T>) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func - (left: Point3<T>, right: Point3<T>) -> Point3<T> {
                return Point3<T>(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }

        public static func - (left: Point3<T>, right: Point3<T>) -> Vector3<T> {
                return Vector3<T>(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }
}



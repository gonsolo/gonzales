public struct FloatMatrix<Backing: MatrixBackingProtocol>: MatrixProtocol {

        public init(t00: Backing.BackingFloat, t01: Backing.BackingFloat, t02: Backing.BackingFloat, t03: Backing.BackingFloat,
                    t10: Backing.BackingFloat, t11: Backing.BackingFloat, t12: Backing.BackingFloat, t13: Backing.BackingFloat,
                    t20: Backing.BackingFloat, t21: Backing.BackingFloat, t22: Backing.BackingFloat, t23: Backing.BackingFloat,
                    t30: Backing.BackingFloat, t31: Backing.BackingFloat, t32: Backing.BackingFloat, t33: Backing.BackingFloat) {

                self.init()

                backing[0, 0] = t00;
                backing[0, 1] = t01;
                backing[0, 2] = t02;
                backing[0, 3] = t03;

                backing[1, 0] = t10;
                backing[1, 1] = t11;
                backing[1, 2] = t12;
                backing[1, 3] = t13;

                backing[2, 0] = t20;
                backing[2, 1] = t21;
                backing[2, 2] = t22;
                backing[2, 3] = t23;

                backing[3, 0] = t30;
                backing[3, 1] = t31;
                backing[3, 2] = t32;
                backing[3, 3] = t33;
        }

        public init(backing: Backing) {
                self.backing = backing
        }


        public init(m: FloatMatrix) {
                self.backing = m.backing
        }

        public init() {
                backing = Backing()
        }

        subscript(row: Int, column: Int) -> Backing.BackingFloat {
                get { return backing[row, column] }
                set { backing[row, column] = newValue }
        }

        public static func * (m1: FloatMatrix, m2: FloatMatrix) -> FloatMatrix {
                var r = FloatMatrix()
                for i in 0..<4 {
                        for j in 0..<4 {
                                let a = m1[i, 0] * m2[0, j]
                                let b = m1[i, 1] * m2[1, j]
                                let c = m1[i, 2] * m2[2, j]
                                let d = m1[i, 3] * m2[3, j]
                                r[i, j] = a + b + c + d
                        }
                }
                return r
        }

        func invert(m: FloatMatrix) -> FloatMatrix {

                var indxc = [0, 0, 0, 0]
                var indxr = [0, 0, 0, 0]
                var ipiv  = [0, 0, 0, 0]
                var minv = backing

                func choosePivot(irow: inout Int, icol: inout Int) throws {
                        var big: Backing.BackingFloat = 0.0
                        for j in 0..<4 {
                                if ipiv[j] != 1 {
                                        for k in 0..<4 {
                                                if ipiv[k] == 0 {
                                                        if abs(minv[j, k]) >= big {
                                                                big = abs(minv[j, k])
                                                                irow = j
                                                                icol = k
                                                        }
                                                } else if ipiv[k] > 1 {
                                                        throw MatrixError.singularMatrix
                                                }
                                        }
                                }
                        }
                }

                func swapColumns() {
                        for j in (0..<4).reversed() {
                                if indxr[j] != indxc[j] {
                                        for k in 0..<4 {
                                                let r = indxr[j]
                                                let c = indxc[j]
                                                let tmp = minv[k, r]
                                                minv[k, r] = minv[k, c]
                                                minv[k, c] = tmp
                                        }
                                }
                        }
                }

                func subtractRow(icol: inout Int) {
                        for j in 0..<4 {
                                if j != icol {
                                        let save = minv[j, icol]
                                        minv[j, icol] = 0
                                        for k in 0..<4 {
                                                minv[j, k] -= minv[icol, k] * save
                                        }
                                }
                        }
                }

                func reduce(_ i: Int) throws {
                        var irow = 0
                        var icol = 0
                        try choosePivot(irow: &irow, icol: &icol)
                        ipiv[icol] = ipiv[icol] + 1
                        if irow != icol {
                                for k in 0..<4 {
                                        let tmp = minv[irow, k]
                                        minv[irow, k] = minv[icol, k]
                                        minv[icol, k] = tmp
                                }
                        }
                        indxr[i] = irow
                        indxc[i] = icol
                        if minv[icol, icol] == 0 {
                                throw MatrixError.singularMatrix
                        }
                        let pivinv = 1 / minv[icol, icol]
                        minv[icol, icol] = 1
                        for j in 0..<4 {
                                minv[icol, j] *= pivinv
                        }
                        subtractRow(icol: &icol)
                }

                do {
                        for i in 0..<4 { try reduce(i) }
                        swapColumns()
                        //return FloatMatrix(minv)
                        return FloatMatrix(backing: minv)
                } catch MatrixError.singularMatrix {
                        warning("Singular matrix encountered!")
                        return FloatMatrix()
                } catch {
                        abort("Unhandled error in matrix invert!")
                }
        }

        public var inverse: FloatMatrix {
                get {
                        return invert(m: self)
                }
        }


        public var isAffine: Bool {
                get {
                        return backing[3, 0] == 0 &&
                               backing[3, 1] == 0 &&
                               backing[3, 2] == 0 &&
                               backing[3, 3] == 1
                }
        }

        var backing: Backing
}

extension FloatMatrix: CustomStringConvertible {
        public var description: String {
                var d = ""
                for i in 0..<4 {
                        d += "[ "
                        for j in 0..<4 {
                                d += "\(backing[i, j])"
                                if j != 3 { d += " " }
                        }
                        d += " ]"
                }
                return d
        }
}


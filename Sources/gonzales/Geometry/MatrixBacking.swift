public struct MatrixBacking<MatrixBackingFloat: BinaryFloatingPoint>: MatrixBackingProtocol {

        public init() {
                m2 = FixedMatrix<MatrixBackingFloat>() 
                m2[0, 0] = 1.0
                m2[1, 1] = 1.0
                m2[2, 2] = 1.0
                m2[3, 3] = 1.0
        }

        public subscript(row: Int, column: Int) -> MatrixBackingFloat {
                get { return m2[row, column] }
                set { m2[row, column] = newValue }
        }

        var m2: FixedMatrix<MatrixBackingFloat>
}


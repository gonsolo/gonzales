/**
        A material provides the look of a surface.
*/
protocol Material: AnyObject {
        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?)
}


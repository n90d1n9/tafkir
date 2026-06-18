package tech.kayys.tafkir.buildlogic

data class DirectFixtureCoverageModuleReport(
    val moduleName: String,
    val fixtures: List<DirectFixtureCoverage>,
)

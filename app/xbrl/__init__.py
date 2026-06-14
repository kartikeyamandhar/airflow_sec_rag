"""XBRL numeric ingestion (the numbers plane).

Parses the EDGAR CompanyFacts JSON acquired in Phase 1 into typed
:class:`~app.domain.models.NumericFact` rows. Every fact keeps its accession, so a
number can be attributed to the filing it came from.
"""

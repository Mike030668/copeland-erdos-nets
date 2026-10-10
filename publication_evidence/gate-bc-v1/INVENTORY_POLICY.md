# Inventory policy

INVENTORY.csv lists all payload files including SHA256SUMS. It excludes itself
to avoid self-reference. SHA256SUMS covers all files except itself and
INVENTORY.csv. Together they enumerate the exhaustive file set; verify set
equality plus raw digests, not only the listed checksums. No temporary files
or Windows metadata are included. SOURCE_TO_PUBLIC covers raw input copies
and derivatives; GENERATED_LINEAGE binds deterministic tables to raw metrics.
Files not present in those mappings are authored packaging/control documents
and are explicitly named in PACKAGE_FILE_ROLES.json.

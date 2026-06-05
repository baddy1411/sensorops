Place machine maintenance manual files here (.txt or .md format).

These are indexed into the RAG store at startup by:
  rag_store.index_manual_directory("data/manuals/")

Example files to add:
  - spindle_maintenance.txt
  - coolant_system_manual.txt
  - tool_wear_guidelines.txt
  - bearing_inspection_procedure.txt

Files are chunked at ~500 characters with 20% overlap.


from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState

class PdfToMdNode(BaseNode):
    name = 'pdf_to_md_node'



"""
导入流程主图
"""
import json

from dotenv import load_dotenv
from langgraph.graph import StateGraph,END

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.knowledge_graph import KnowLedgeGraphNode
from knowledge.processor.import_process.nodes.md_img import MdImgNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState

load_dotenv()

def import_router(state:ImportGraphState) -> str:
    """
        入口节点后的路由逻辑

        根据文件类型决定走 PDF 转换分支还是直接处理 MD 分支

        Args:
            state: 当前图状态

        Returns:
            下一个节点名称
    """
    if state.get("is_md_read_enabled"):
        return "md_img"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md"
    return END

def create_import_graph():
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry
          │
          ├── (PDF) ──> pdf_to_md ──┐
          │                         │
          └── (MD) ────────────────>├──> md_img
                                    │
                                    v
                            document_split
                                    │
                                    v
                        item_name_recognition
                                    │
                                    v
                            bge_embedding
                                    │
                                    v
                            import_milvus
                                    │
                                    v
                          knowledge_graph
                                    │
                                    v
                                   END
    """
    graph_pineline=StateGraph(ImportGraphState)
    graph_pineline.set_entry_point('entry_node')

    nodes={
        'entry_node':EntryNode(),
        'pdf_to_md_node':PdfToMdNode(),
        'md_img_node':MdImgNode(),
        'document_split_node':DocumentSplitNode(),
        'item_name_recognition_node':ItemNameRecognitionNode(),
        'beg_embedding_node':BgeEmbeddingNode(),
        'import_milvus_node':ImportMilvusNode(),
        'kg_node':KnowLedgeGraphNode()
    }
    for key,value in nodes.items():
        graph_pineline.add_node(key,value)

    graph_pineline.add_conditional_edges('entry_node',
                                         import_router,
                                         {
                                             'md_img':'md_img_node',
                                             'pdf_to_md':'pdf_to_md_node',
                                             END:END
                                         })
    graph_pineline.add_edge('pdf_to_md_node','md_img_node')
    graph_pineline.add_edge('md_img_node','document_split_node')
    graph_pineline.add_edge('document_split_node','item_name_recognition_node')
    graph_pineline.add_edge('item_name_recognition_node','beg_embedding_node')
    graph_pineline.add_edge('beg_embedding_node','import_milvus_node')
    graph_pineline.add_edge('import_milvus_node','kg_node')
    graph_pineline.add_edge('kg_node',END)
    return graph_pineline.compile()
kb_import__graph_app=create_import_graph()

if __name__ == '__main__':
    setup_logging()

    import_file_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf"
    file_dir = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir"
    # 1. 测试编排流程
    final_state = run_import_graph(import_file_path=import_file_path, file_dir=file_dir)
    print(json.dumps(final_state, indent=2, ensure_ascii=False))

    # 2.打印图结构（ASCII 可视化）# 1. 单独安装：pip install grandalf 2.(单独安装还出错)  【pydantic：定义数据模型 】pip uninstall gradio  3. 单独安装 pip install grandalf 解决冲突
    print("-" * 50)
    print("图结构:")
    kb_import__graph_app.get_graph().print_ascii()
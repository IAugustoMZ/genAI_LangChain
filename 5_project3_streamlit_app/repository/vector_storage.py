import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from infra.open_ai_tools import OpenAITools
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# define the path to the .env file
env_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    ), 'env', '.env'
)
load_dotenv(dotenv_path=env_path)

class ChromaVectorStorage:

    def __init__(self):
        """
        class that manages the vector storage
        """
        self.openai = OpenAITools()

    @staticmethod
    def load_document(file_path) -> str:
        """
        Load the document from the file path

        Parameters:
        ----------
        file_path: str
            The path to the document
        
        Returns:
        -------
        str
            The document string
        """

        # check if the document is a pdf
        if file_path.endswith('.pdf'):

            # instantiate the document loader
            loader = PyPDFLoader(file_path=file_path)
        else:
            raise ValueError("The document file type is not supported")

        # load the document
        document = loader.load()

        return document
    
    @staticmethod
    def chunk_data(data: str, chunk_size: int=256, chunk_overlap: int=0):
        """
        Chunk the data into smaller pieces

        Parameters:
        ----------
        data: str
            The data to be chunked
        chunk_size: int
            The size of the chunks
        chunk_overlap: int
            The overlap between the chunks
        
        Returns:
        -------
        list
            The list of chunks
        """

        # instantiate the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # split the data into chunks
        chunks = text_splitter.split_documents(data)
    
        return chunks
    

    def create_embedding_store(self,
                               chunks: list,
                               persist_directory: str='repository/data/chroma.db') -> object:
        """
        Create the embedding store

        Parameters
        ----------
        chunks : list
            The list of chunks
        persist_directory : str, optional
            directory to persist the chroma db, by default 'repository/data/chroma.db'

        Returns
        -------
        object
            The vector store object
        """

        # create vector score
        vector_store = Chroma.from_documents(
            chunks,
            embedding=self.openai.embeddings,
            persist_directory=persist_directory
        )

        return vector_store
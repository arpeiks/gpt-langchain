import "pdf-parse";
import "dotenv/config";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

const run = async () => {
  const directoryLoader = new DirectoryLoader("docs", {
    ".pdf": (path) => new PDFLoader(path),
  });

  const rawDocs = await directoryLoader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const docs = await textSplitter.splitDocuments(rawDocs);

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index("demo");

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  await PineconeStore.fromDocuments(docs, embeddings, {
    namespace: "test-resume",
    pineconeIndex: pineconeIndex,
  });
};

(async () => {
  await run();
  console.log("ingestion complete");
})();

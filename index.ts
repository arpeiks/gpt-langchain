import "dotenv/config";
import { makechain } from "./makechain";
import { PineconeStore } from "@langchain/pinecone";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";

const history: [string, string][] = [];
const question = "Give all relevant contact details of the owner";

const run = async () => {
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index("demo");

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    namespace: "test-resume",
    pineconeIndex: pineconeIndex,
  });

  let resolveWithDocuments: (value: Document[]) => void;

  const documentPromise = new Promise<Document[]>((resolve) => {
    resolveWithDocuments = resolve;
  });

  const retriever = vectorStore.asRetriever({
    callbacks: [
      {
        handleRetrieverEnd(documents) {
          resolveWithDocuments(documents);
        },
      },
    ],
  });

  const chain = makechain(retriever);

  const pastMessages = history
    .map((message: [string, string]) => {
      return [`Human: ${message[0]}`, `Assistant: ${message[1]}`].join("\n");
    })
    .join("\n");

  const response = await chain.invoke({
    question,
    chat_history: pastMessages,
  });

  const sourceDocuments = await documentPromise;

  console.log({ text: response, sourceDocuments });
  return { text: response, sourceDocuments };
};

(async () => {
  await run();
  console.log("ingestion complete");
})();

import { loadQARefineChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import express from "express";

async function runQA(query) {
    // Create the models and chain
    const embeddings = new OpenAIEmbeddings({ openAIApiKey: "YOUR_OPENAI_API_KEY" });
    const model = new OpenAI({ openAIApiKey: "YOUR_OPENAI_API_KEY" });
    const chain = loadQARefineChain(model);

    // Load the documents and create the vector store
    const loader = new PDFLoader("./example.pdf");
    const docs = await loader.loadAndSplit();
    const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

    // Select the relevant documents
    const question = query;
    const relevantDocs = await store.similaritySearch(question);

    // Call the chain
    const res = await chain.call({
        input_documents: relevantDocs,
        question,
    });

    return res;
    /*
    {
      output_text: '\n' +
        '\n' +
        "answer"
    }
    */
}

//setup the server and routes

const app = express();
const port = 3000;

app.get("/qa", async (req, res) => {
    try {
        const {query} = req.query
        const answer = await runQA(query);
        res.status(200).send(answer);

    } catch (err) {
        console.log(err);
        res.status(500).send(err);
    }
});

app.listen(3000, () => console.log("Server started on " + port));

//run server and test api by sending GET requests like following:
//http://localhost:3000/qa?query="question?"
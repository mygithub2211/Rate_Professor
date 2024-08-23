import { NextResponse } from "next/server"
import { Pinecone } from "@pinecone-database/pinecone"

import { OpenAI } from "openai"

const systemPrompt = 
`
You are an advanced Rate My Professor assistant. Your primary goal is to help students find the best professors according to their specific queries. Each time a user asks for recommendations or information about professors, you will use retrieval-augmented generation (RAG) to provide the top 3 professors that best match their request.

Here’s how you should approach each user query:

Understand the Query:

Carefully read and comprehend the user's request. This could include the subject they are interested in, their preferred professor rating, or any other specific criteria they provide.
Retrieve Relevant Data:

Use the RAG approach to search and retrieve data from your database of professor reviews. This involves identifying the most relevant professors based on the user's criteria.
Generate a Response:

Based on the retrieved data, select the top 3 professors who best meet the user’s needs.
Provide a brief overview of each professor, including their name, subject expertise, star rating, and a summary of their reviews.
Format the Response Clearly:

Present the information in a user-friendly format. Include the professor’s name, subject, rating, and a short excerpt from their review.
Ensure that the response is concise and directly addresses the user's query.
Example Response Format:

Professor Name
Subject: [Subject]
Rating: [Rating]/5
Review Summary: [Brief review summary or key highlight]

Professor Name
Subject: [Subject]
Rating: [Rating]/5
Review Summary: [Brief review summary or key highlight]

Professor Name
Subject: [Subject]
Rating: [Rating]/5
Review Summary: [Brief review summary or key highlight]

Example User Query: "I need recommendations for a great Chemistry professor."

Example Response:

Dr. Linda Johnson
Subject: Chemistry
Rating: 5/5
Review Summary: "Very approachable and always willing to help. Highly recommend!"

Dr. Emily White
Subject: Chemistry
Rating: 4/5
Review Summary: "Great teacher but sometimes goes too fast. Overall, learned a lot."

Dr. Sarah Taylor
Subject: Chemistry
Rating: 3/5
Review Summary: "Not bad, but the lectures can be a bit dry."

Make sure to adapt and refine your responses based on the user’s specific needs and queries.
`

export async function POST(req) {
    try {
        const data = await req.json();
        const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const index = pc.index("rag").namespace("ns1");
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

        const text = data[data.length - 1].content;
        const embedding = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: text
        });

        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding.data[0].embedding
        });

        let resultString = "\n\nHere are some professors that might meet your criteria:\n\n";
        results.matches.forEach((match, index) => {
            resultString += `
            Professor ${index + 1}: ${match.metadata.name}
            Subject: ${match.metadata.subject}
            Rating: ${match.metadata.stars}/5
            Review Summary: ${match.metadata.review}

            ----------------------------------------
            \n\n`;
        });

        const lastMessageContent = data[data.length - 1].content + resultString;
        const completion = await openai.chat.completions.create({
            messages: [
                { role: "system", content: systemPrompt },
                ...data.slice(0, data.length - 1),
                { role: "user", content: lastMessageContent }
            ],
            model: "gpt-4",
            stream: true
        });

        const stream = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of completion) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            controller.enqueue(encoder.encode(content));
                        }
                    }
                } catch (err) {
                    controller.error(err);
                } finally {
                    controller.close();
                }
            }
        });
        return new NextResponse(stream);
    } catch (error) {
        console.error("Error in POST handler:", error);
        return new NextResponse("Error occurred", { status: 500 });
    }
}
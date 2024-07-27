import pytest
import pandas as pd
from ast import literal_eval
from embeddings import (
    Document,
    DocumentChunk,
    create_document_chunks,
)

CHUNK_SIZE = 200
DOC_ID = "20e01e08-12bd-4258-ab45-5cf9244b727f"


@pytest.fixture
def document() -> Document:
    text = """Executives at Microsoft and OpenAI have been drawing up plans for a data center project that would contain a supercomputer with millions of specialized server chips to power OpenAI’s artificial intelligence, according to three people who have been involved in the private conversations about the proposal. The project could cost as much as $100 billion, according to a person who spoke to OpenAI CEO Sam Altman about it and a person who has viewed some of Microsoft’s initial cost estimates. Microsoft would likely be responsible for financing the project, which would be 100 times more costly than some of today’s biggest data centers, demonstrating the enormous investment that may be needed to build computing capacity for AI in the coming years. Executives envisage the proposed U.S.-based supercomputer, which they have referred to as “Stargate,” as the biggest of a series of installations the companies are looking to build over the next six years. The Takeaway • Microsoft executives are looking to launch Stargate as soon as 2028 • The supercomputer would require an unprecedented amount of power • OpenAI’s next major AI upgrade is expected to land by early next year While project has not been green-lit and the plans could change, they provide a peek into this decade’s most important tech industry tie-up and how far ahead the two companies are thinking. Microsoft so far has committed more than $13 billion to OpenAI so the startup can use Microsoft data centers to power ChatGPT and the models behind its conversational AI. In exchange, Microsoft gets access to the secret sauce of OpenAI’s technology and the exclusive right to resell that tech to its own cloud customers, such as Morgan Stanley. Microsoft also has baked OpenAI’s software into new AI Copilot features for Office, Teams and Bing. Microsoft’s willingness to go ahead with the Stargate plan depends in part on OpenAI’s ability to meaningfully improve the capabilities of its AI, one of these people said. OpenAI last year failed to deliver a new model it had promised to Microsoft, showing how difficult the AI frontier can be to predict. Still, OpenAI CEO Sam Altman has said publicly that the main bottleneck holding up better AI is a lack of sufficient servers to develop it. If Stargate moves forward, it would produce orders of magnitude more computing power than what Microsoft currently supplies to OpenAI from data centers in Phoenix and elsewhere, these people said. The proposed supercomputer would also require at least several gigawatts of power—equivalent to what’s needed to run at least several large data centers today, according to two of these people. Much of the project cost would lie in procuring the chips, two of the people said, but acquiring enough energy sources to run it could also be a challenge. Such a project is “absolutely required” for artificial general intelligence—AI that can accomplish most of the computing tasks humans do, said Chris Sharp, chief technology officer of Digital Realty, a data center operator that hasn’t been involved in Stargate. Though the project’s scale seems unimaginable by today’s standard, he said that by the time such a supercomputer is finished, the numbers won’t seem as eye-popping. A Microsoft data center near Phoenix that isn't related to OpenAI. Image via Microsoft The executives have discussed launching Stargate as soon as 2028 and expanding it through 2030, possibly needing as much as 5 gigawatts of power by the end, the people involved in the discussions said. Phase Five Altman and Microsoft employees have talked about these supercomputers in terms of five phases, with phase 5 being Stargate, named for a science fiction film in which scientists develop a device for traveling between galaxies. (The codename originated with OpenAI but isn’t the official project codename that Microsoft is using, said one person who has been involved.) The phase prior to Stargate would cost far less. Microsoft is working on a smaller, phase 4 supercomputer for OpenAI that it aims to launch around 2026, according to two of the people. Executives have planned to build it in Mt. Pleasant, Wisc., where the Wisconsin Economic Development Corporation recently said Microsoft broke ground on a $1 billion data center expansion. The supercomputer and data center could eventually cost as much as $10 billion to complete, one of these people said. That’s many times more than the cost of existing data centers. Microsoft also has discussed using Nvidia-made AI chips for that project, said a different person who has been involved in the conversations. Today, Microsoft and OpenAI are in the middle of phase 3 of the five-phase plan. Much of the cost of the next two phases will involve procuring the AI chips. Two data center practitioners who aren’t involved in the project said it’s common for AI server chips to make up around half of the total initial cost of AI-focused data centers other companies are currently building. All up, the proposed efforts could cost in excess of $115 billion, more than three times what Microsoft spent last year on capital expenditures for servers, buildings and other equipment. Microsoft was on pace to spend around $50 billion this year, assuming it continues the pace of capital expenditures it disclosed in the second half of 2023. Microsoft CFO Amy Hood said in January that such spending will increase “materially” in the coming quarters, driven by investments in “cloud and AI infrastructure.” Frank Shaw, a Microsoft spokesperson, did not comment about the supercomputing plans but said in a statement: “We are always planning for the next generation of infrastructure innovations needed to continue pushing the frontier of AI capability.” An OpenAI spokesperson did not have a comment for this article. Altman has said privately that Google, one of OpenAI’s biggest rivals, will have more computing capacity than OpenAI in the near term, and publicly he has complained about not having as many AI server chips as he’d like. That’s one reason he has been pitching the idea of a new server chip company that would develop a chip rivaling Nvidia’s graphics processing unit, which today powers OpenAI’s software. Demand for Nvidia GPU servers has skyrocketed, driving up costs for customers such as Microsoft and OpenAI. Besides controlling costs, Microsoft has other potential reasons to support Altman’s alternative chip. The GPU boom has put Nvidia in the position of kingmaker as it decides which customers can have the most chips, and it has aided small cloud providers that compete with Microsoft. Nvidia has also muscled into reselling cloud servers to its own customers. With or without Microsoft, Altman’s effort would require significant investments in power and data centers to accompany the chips. Stargate is designed to give Microsoft and OpenAI the option of using GPUs made by companies other than Nvidia, such as Advanced Micro Devices, or even an AI server chip Microsoft recently launched, said the people who have been involved in the discussions. It isn’t clear whether Altman believes the theoretical GPUs he aims to develop in the coming years will be ready for Stargate. The total cost of the Stargate supercomputer could depend on software and hardware improvements that make data centers more efficient over time. The companies have discussed the possibility of using alternative power sources, such as nuclear energy, according to one of the people involved. (Amazon just purchased a Pennsylvania data center site with access to nuclear power. Microsoft also had discussed bidding on the site, according to two people involved in the talks.) Altman himself has said that developing superintelligence will likely require a significant energy breakthrough. Packed Racks To make Stargate a reality, Microsoft also would have to overcome several technical challenges, the two people said. For instance, the current proposed design calls for putting many more GPUs into a single rack than Microsoft is used to, to increase the chips’ efficiency and performance. Because of the higher density of GPUs, Microsoft would also need to come up with a way to prevent the chips from overheating, they said. Microsoft and OpenAI are also debating which cables they will use to string the millions of GPUs together. The networking cables are crucial for moving large amounts of data in and out of server chips quickly. OpenAI has told Microsoft it doesn’t want to use Nvidia’s proprietary InfiniBand cables in the Stargate supercomputer, even though Microsoft currently uses the Nvidia cables in its existing supercomputers, according to two people who were involved in the discussions. (OpenAI instead wants to use more generic Ethernet cables.) Switching away from InfiniBand could make it easier for OpenAI and Microsoft to lessen their reliance on Nvidia down the line. AI computing is more expensive and complex than traditional computing, which is why companies closely guard the details about their AI data centers, including how GPUs are connected and cooled. For his part, Nvidia CEO Jensen Huang has said companies and countries will need to build $1 trillion worth of new data centers in the next four to five years to handle all of the AI computing that’s coming. Microsoft and OpenAI executives have been discussing the data center project since at least last summer. Besides CEO Satya Nadella and Chief Technology Officer Kevin Scott, other Microsoft managers who have been involved in the supercomputer talks have included Pradeep Sindhu, who leads strategy for the way Microsoft stitches together AI server chips in its data centers, and Brian Harry, who helps develop AI hardware for the Azure cloud server unit, according to people who have worked with them. OpenAI President Greg Brockman, left, and Microsoft CTO Kevin Scott. Photo via YouTube/Microsoft Developer The partners are still ironing out several key details, which they might not finalize anytime soon. It is unclear where the supercomputer will be physically located and whether it will be built inside one data center or multiple data centers in close proximity. Clusters of GPUs tend to work more efficiently when they are located in the same data center. OpenAI has already pushed the boundaries of what Microsoft can do with data centers. After making its initial investment in the startup in 2019, Microsoft built its first GPU supercomputer, containing thousands of Nvidia GPUs, to handle OpenAI’s computing demands, spending $1.2 billion on the system over several years. This year and next year, Microsoft has planned to provide OpenAI with servers housing hundreds of thousands of GPUs in total, said a person with knowledge of its computing needs. The Next Barometer: GPT-5 Microsoft and OpenAI’s grand designs for world-beating data centers depend almost entirely on whether OpenAI can help Microsoft justify the investment in those projects by taking major strides toward superintelligence—AI that can help solve complex problems such as cancer, fusion, global warming or colonizing Mars. Such attainments may be a far-off dream. While some consumers and professionals have embraced ChatGPT and other conversational AI as well as AI-generated video, turning these recent breakthroughs into technology that produces significant revenue could take longer than practitioners in the field anticipated. Firms including Amazon and Google have quietly tempered expectations for sales, in part because such AI is costly and requires a lot of work to launch inside large enterprises or to power new features in apps used by millions of people. Altman said at an Intel event last month that AI models get “predictably better” when researchers throw more computing power at them. OpenAI has published research on this topic, which it refers to as the “scaling laws” of conversational AI. OpenAI “throwing ever more compute [power to scale up existing AI] risks leading to a ‘trough of disillusionment’” among customers as they realize the limits of the technology, said Ali Ghodsi, CEO of Databricks, which helps companies use AI. “We should really focus on making this technology useful for humans and enterprises. That takes time. I believe it’ll be amazing, but [it] doesn’t happen overnight.” The stakes are high for OpenAI to prove that its next major conversational AI, known as a large language model, is significantly better than GPT-4, its most advanced LLM today. OpenAI released GPT-4 a year ago, and Google has released a comparable model in the meantime as it tries to catch up. OpenAI aims to release its next major LLM upgrade by early next year, said one person with knowledge of the process. It could release more incremental improvements to LLMs before then, this person said. With more servers available, some OpenAI leaders believe the company can use its existing AI and recent technical breakthroughs such as Q*—a model that can reason about math problems it hasn’t previously been trained to solve—to create the right synthetic (non–human-generated) data for training better models after running out of human-generated data to give them. These models may also be able to figure out the flaws in existing models like GPT-4 and suggest technical improvements—in other words, self-improving AI."""
    return Document(text=text, id=DOC_ID)


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    df = pd.read_csv(f"data/{DOC_ID}.csv")
    df["embedding"] = df["embedding"].apply(literal_eval)
    chunks = []
    for i, row in df.iterrows():
        chunk = DocumentChunk(
            index=i,
            doc_id=DOC_ID,
            text=row["text"],
            embedding=row["embedding"],
        )
        chunks.append(chunk)
    return chunks


def test_create_document_chunks(document: Document, chunks: list[DocumentChunk]):
    doc_chunks, doc_id = create_document_chunks(document, CHUNK_SIZE)
    assert doc_id == DOC_ID
    assert len(doc_chunks) == len(chunks)

    for generated_chunk, expected_chunk in zip(doc_chunks, chunks):
        assert generated_chunk.index == expected_chunk.index
        assert generated_chunk.doc_id == expected_chunk.doc_id
        assert generated_chunk.text == expected_chunk.text

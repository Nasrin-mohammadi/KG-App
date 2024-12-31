import streamlit as st
import os

st.title("KG Construction App")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
else:
    st.stop()
from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import Document
from tempfile import NamedTemporaryFile

output_parser = CommaSeparatedListOutputParser()
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from tempfile import NamedTemporaryFile
# Initialize objects
embeddings = OpenAIEmbeddings()
output_parser = CommaSeparatedListOutputParser()


msgs = ChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

def process_pdf(pdf_path):
    all_results1 = []
    all_results2 = []
    all_results = []

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()

    big_chunks_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    msgs = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    template1 =""" 

You are a Dental Health assistant specializing in extracting entities and relationships for knowledge graphs using the <http://example.com/dental#> ontology. Your expertise lies in dental health records and patient care workflows. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.

Use the unique ID for any newly created entities. Extract the nodes based on these definitions:

- Velscope: A non-invasive diagnostic tool used for detecting oral abnormalities, including early signs of oral cancer, through fluorescence visualization.

- Medical History Update (HX): A record of any changes or updates in the patient’s medical history since their last visit, including allergies, medications, and systemic conditions.

- Smoker: An attribute indicating whether the patient uses tobacco products, which is a critical factor in assessing oral health risks.

- Dental Hygiene Diagnosis (DH): The identification and documentation of oral health conditions based on a hygienist's assessment.

- Chief Complaint (CC): The primary issue or concern expressed by the patient during their visit.

- Extra Oral/Intra Oral Exam (EOIO): A comprehensive examination of the head, neck, and oral cavity to identify abnormalities or conditions affecting oral and overall health.

- Dentition: The arrangement, condition, and alignment of the teeth in the oral cavity.

- Sensitivity: A condition where teeth or gums react to certain stimuli, such as temperature or pressure.

- Deposits: Accumulated plaque, tartar, or other substances on teeth and gums that can lead to oral health issues.

- Bleeding: The presence of blood in the gums, often an indicator of gingivitis or periodontal disease.

- Perio Status: The health status of the periodontal tissues, including gums and supporting bone, assessed during periodontal evaluation.

- Oral Health Instructions (OHI): Personalized guidance provided to the patient on maintaining oral hygiene, including brushing and flossing techniques.

- Hygiene Treatment (HTX): Procedures performed by a dental hygienist to improve oral health, such as scaling, polishing, and debridement.

- Re-care Exam (RC): A routine follow-up examination to monitor and maintain the patient’s oral health over time.

- Next Visit (NV): The planned date and purpose of the patient’s subsequent dental appointment.

**node extraction rules**

Follow this structure for extracting the entities:

- Look for any example of dental entity that should be defined as nodes in the knowledge graph with minimal details necessary for understanding the relationship for the knowledge graph.
- Dental entities include diagnostic tools, procedures, patient attributes, assessments, and treatment records.
- If an attribute has multiple distinct values, each value should be represented by a separate attribute node in the knowledge graph.
- If different patients or contexts share the same attribute (e.g., chief complaint), each instance should be represented as a separate node in the knowledge graph.
- The value of any quantifiable attribute (e.g., Perio Status) must be numeric and accompanied by a unit (if applicable) and cannot be a string or text alone.
- Procedures (e.g., hygiene treatment, re-care exam) should be represented as separate nodes, capturing only their core purpose and context.
- Always represent each unique entity distinctly to avoid loss of granularity in the knowledge graph.

ALWAYS REMEMBER:

- Extract all the nodes that should be defined in the knowledge graph with minimal details necessary for understanding the relationships.
- Always remember that nodes are newly created entities that should be extracted from the text.
- It is critical to extract all the nodes, as you are a dental health expert who knows the importance of capturing all diagnostic tools, procedures, patient attributes, assessments, and treatments in the graph.
- Focus solely on extracting nodes for the knowledge graph, such as 'Velscope', 'Medical History Update (HX)', 'Smoker', 'Dental Hygiene Diagnosis (DH)', 'Chief Complaint (CC)', 'Extra Oral/Intra Oral Exam (EOIO)', 'Dentition', 'Sensitivity', 'Deposits', 'Bleeding', 'Perio Status', 'Oral Health Instructions (OHI)', 'Hygiene Treatment (HTX)', 'Re-care Exam (RC)', and 'Next Visit (NV)'.
- Always double-check to extract all the entities needed for creating relationships in the knowledge graph.
- Avoid extra explanations; directly format the output as:

"Velscope": [] # it can be a list  
"Medical History Update (HX)": [] # it can be a list  
"Smoker": [] # it can be a list  
"Dental Hygiene Diagnosis (DH)": [] # it can be a list  
"Chief Complaint (CC)": [] # it can be a list  
"Extra Oral/Intra Oral Exam (EOIO)": [] # it can be a list  
"Dentition": [] # it can be a list  
"Sensitivity": [] # it can be a list  
"Deposits": [] # it can be a list  
"Bleeding": [] # it can be a list  
"Perio Status": [] # it can be a list  
"Oral Health Instructions (OHI)": [] # it can be a list  
"Hygiene Treatment (HTX)": [] # it can be a list  
"Re-care Exam (RC)": [] # it can be a list  
"Next Visit (NV)": [] # it can be a list  

** Here is an Example to help you how extract the nodes from a text. you should NEVER use the context of this examples to answer the questions, because these are just examples to help you to better understand the task.

- Example for Extracting nodes from this text in Angle brackets:

< Velscope: Velscope oral cancer screening performed; results were normal.
 Medical History Update (HX): Patient takes Trazodone 50 mg. No allergies reported.
 Smoker: No.
 Dental Hygiene Diagnosis (DH): Generalized slight to moderate plaque and calculus. Generalized 1-3 mm pockets. 
 Chief Complaint (CC): Food impaction in back molars.
 Extra Oral/Intra Oral Exam (EOIO): Lymph nodes normal and non-palpable. Range of motion: 45 mm opening.
 Dentition: Missing all third molars (wisdom teeth) and all first premolars. 
 Sensitivity: No cold sensitivity reported.
 Deposits: Very little build-up observed.
 Bleeding: Gums do not bleed.
 Perio Status: Generalized 1-3 mm periodontal pockets. Localized 4 mm pockets in quadrant three posterior region. Generalized 2-3 mm gingival recession.
 Oral Health Instructions (OHI): Encouraged continued excellent oral hygiene practices.
 Hygiene Treatment (HTX): Performed dental cleaning and polishing. Fluoride rinse administered.
 Re-care Exam (RC): Next cleaning scheduled in six months.
 Next Visit (NV): Plan to restore teeth 36 and 37 with crowns.Monitoring of tooth 46 apical lesion. >

 in this example: Velscope is 'normal',  Medical History Update (HX) is 'Trazodone, value: 50 mg' and 'alergies, value: No', Smoker is 'No', 
 Dental Hygiene Diagnosis (DH) is 'slight to moderate plaque and calculus' and 'pockets, value:1-3 mm', Chief Complaint (CC) is 'Food impaction in back molars', 
 Extra Oral/Intra Oral Exam (EOIO) is 'Lymph nodes, value:normal and non-palpable' and  'Range of motion, value:45 mm', 
 Dentition is  'Missing all wisdom teeth' and 'missing all first premolars', Sensitivity is  'No cold sensitivity',  Deposits is 'Very little build-up', 
Bleeding is 'No',  Perio Status is  'periodontal pockets, value:1-3 mm' and 'Localized pockets, value:4 mm' and 'Generalized gingival recession, value:2-3 mm',
Oral Health Instructions (OHI) is 'continued excellent oral hygiene practices',  Hygiene Treatment (HTX) is 'dental cleaning' and 'polishing' and 'Fluoride rinse administered''
Re-care Exam (RC) is  'in six months',  Next Visit (NV) is 'restore teeth 36 and 37 with crowns' and 'Monitoring of tooth 46 apical lesion' >


{chat_history}
{context}
Question: {question}
."""
    prompt1 = PromptTemplate(
        template=template1, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )

    template2 = """ You are a Dental Medical assistant specializing in extracting entities and relationships within knowledge graphs for dental diagnosis and treatment workflows. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.
Extract the relationships between the nodes based on these definitions:

"is_diagnosed_with": signifies the condition or status (e.g., smoker, perio status, sensitivity) identified during the diagnostic or examination process.
Domain: Medical History Update (HX), Extra Oral/Intra Oral Exam (EOIO), Dental Hygiene Diagnosis (DH)
Range: Chief Complaint (CC), Perio Status, Sensitivity

"is_assessed_with": denotes the diagnostic tool or process used to evaluate a patient's oral health or specific complaint.
Domain: Chief Complaint (CC), Perio Status, Sensitivity
Range: Velscope, Extra Oral/Intra Oral Exam (EOIO)

"requires_treatment": refers to the treatment or hygiene process recommended based on the diagnosis or assessment.
Domain: Chief Complaint (CC), Perio Status, Sensitivity, Dental Hygiene Diagnosis (DH)
Range: Hygiene Treatment (HTX), Oral Health Instructions (OHI)

"is_followed_by": represents the relationship between a treatment or hygiene process and subsequent examinations or follow-ups.
Domain: Hygiene Treatment (HTX), Oral Health Instructions (OHI)
Range: Re-care Exam (RC), Next Visit (NV)

"has_observation": signifies the specific findings or observations (e.g., deposits, bleeding) recorded during the diagnostic process.
Domain: Extra Oral/Intra Oral Exam (EOIO), Dental Hygiene Diagnosis (DH)
Range: Deposits, Bleeding

"documents": represents the relationship where patient records or updates are linked to specific observations, treatments, or diagnoses.
Domain: Medical History Update (HX), Re-care Exam (RC)
Range: Chief Complaint (CC), Perio Status, Sensitivity, Hygiene Treatment (HTX), Oral Health Instructions (OHI)

ALWAYS REMEMBER:

- Focus solely on extracting relationships between nodes.
- If new nodes are identified during relationship extraction, include them in the output with minimal details necessary for understanding the relationship.
- Avoid extra explanations; directly format the output as:

"is_diagnosed_with": [] # it can be a list  
"is_assessed_with": [] # it can be a list  
"requires_treatment": [] # it can be a list  
"is_followed_by": [] # it can be a list  
"has_observation": [] # it can be a list  
"documents": [] # it can be a list  

**Relationships extraction rules**
As you contribute to building the knowledge graph, following the established rules is crucial. These rules ensure that the graph accurately represents the dental medical domain, maintains structural integrity, and adheres to the established ontology. Please read and apply the following guidelines carefully when extracting the relationships:
- Every node must have at least one edge connecting it to another node.
- Pair diagnostic and treatment nodes with observation nodes using "has_observation" edges.
- NEVER CONNECT one observation node with more than one diagnostic node.
- Every follow-up node (e.g., Re-care Exam, Next Visit) must have at least one edge to either a treatment or diagnostic node.
- Each condition node can only share a "documents" edge with one patient record node (e.g., Medical History Update).
- Each triple should follow the format: (subject, predicate, object).
- Subjects and objects of the triples are the nodes extracted and stored in the history.
- Avoid ANY extra explanation about nodes and relationships; JUST give the output in the specified format, without additional sentences or context.
- If asked, extract only the specified relationship, NOT all relationships.
- If you have extracted the nodes and relationships in the specified format, you do not need to explain each relationship separately.

{chat_history}
{context}
Question: {question}


."""

    prompt2 = PromptTemplate(
        template=template2, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )

    template3 =""" You are a Dental Medical assistant specializing in extracting nodes and edges for knowledge graphs using a customized ontology formatted as TTL. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.

Use the prefix "ex:" with IRI <http://example.com/> for any newly created entities stored in chat history with a unique ID. Only utilize classes defined in this ontology.

@prefix : <http://example.com/dental#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@base <http://example.com/dental#> .

#################################################################
# Classes
#################################################################

###  http://example.com/dental#Velscope
:Velscope rdf:type owl:Class ;
          skos:prefLabel "Velscope" .

###  http://example.com/dental#Medical_History_Update
:Medical_History_Update rdf:type owl:Class ;
                        skos:prefLabel "Medical History Update (HX)" .

###  http://example.com/dental#Smoker
:Smoker rdf:type owl:Class ;
        skos:prefLabel "Smoker" .

###  http://example.com/dental#Dental_Hygiene_Diagnosis
:Dental_Hygiene_Diagnosis rdf:type owl:Class ;
                          skos:prefLabel "Dental Hygiene Diagnosis (DH)" .

###  http://example.com/dental#Chief_Complaint
:Chief_Complaint rdf:type owl:Class ;
                 skos:prefLabel "Chief Complaint (CC)" .

###  http://example.com/dental#Extra_Oral_Intra_Oral_Exam
:Extra_Oral_Intra_Oral_Exam rdf:type owl:Class ;
                            skos:prefLabel "Extra Oral/Intra Oral Exam (EOIO)" .

###  http://example.com/dental#Dentition
:Dentition rdf:type owl:Class ;
           skos:prefLabel "Dentition" .

###  http://example.com/dental#Sensitivity
:Sensitivity rdf:type owl:Class ;
             skos:prefLabel "Sensitivity" .

###  http://example.com/dental#Deposits
:Deposits rdf:type owl:Class ;
          skos:prefLabel "Deposits" .

###  http://example.com/dental#Bleeding
:Bleeding rdf:type owl:Class ;
          skos:prefLabel "Bleeding" .

###  http://example.com/dental#Perio_Status
:Perio_Status rdf:type owl:Class ;
              skos:prefLabel "Perio Status" .

###  http://example.com/dental#Oral_Health_Instructions
:Oral_Health_Instructions rdf:type owl:Class ;
                          skos:prefLabel "Oral Health Instructions (OHI)" .

###  http://example.com/dental#Hygiene_Treatment
:Hygiene_Treatment rdf:type owl:Class ;
                   skos:prefLabel "Hygiene Treatment (HTX)" .

###  http://example.com/dental#Re_care_Exam
:Re_care_Exam rdf:type owl:Class ;
              skos:prefLabel "Re-care Exam (RC)" .

###  http://example.com/dental#Next_Visit
:Next_Visit rdf:type owl:Class ;
            skos:prefLabel "Next Visit (NV)" .

#################################################################
# Object Properties
#################################################################
###  http://example.com/is_diagnosed_with

:is_diagnosed_with rdf:type owl:ObjectProperty ;
                   rdfs:domain :Diagnostic ;
                   rdfs:range :Condition ;
                   skos:prefLabel "is diagnosed with" ;
                   rdfs:comment "Links a diagnostic procedure to the condition it identifies." .

###  http://example.com/is_assessed_with

:is_assessed_with rdf:type owl:ObjectProperty ;
                  rdfs:domain :Condition ;
                  rdfs:range :AssessmentTool ;
                  skos:prefLabel "is assessed with" ;
                  rdfs:comment "Links a condition to the assessment tool used for evaluation." .

###  http://example.com/has_observation

:has_observation rdf:type owl:ObjectProperty ;
                 rdfs:domain :Diagnostic ;
                 rdfs:range :Observation ;
                 skos:prefLabel "has observation" ;
                 rdfs:comment "Links a diagnostic procedure to the observations made." .

###  http://example.com/requires_treatment

:requires_treatment rdf:type owl:ObjectProperty ;
                    rdfs:domain :Condition ;
                    rdfs:range :Treatment ;
                    skos:prefLabel "requires treatment" ;
                    rdfs:comment "Links a condition to its recommended treatment." .

###  http://example.com/documents

:documents rdf:type owl:ObjectProperty ;
           rdfs:domain :PatientRecord ;
           rdfs:range :Condition ;
           skos:prefLabel "documents" ;
           rdfs:comment "Links a patient record to the documented condition." .

** Here is an Example to help you how extract the json-ld. you should NEVER use the context of this examples to answer the questions, because these are just examples to help you to better understand the task.

  "@context" required just "ex": "http://example.com/dental#" and no more explanation, because of the context winodow limitation of LLMs.

  "@graph" for each node class needed to be included id, type, skos and status like this example:
      "@id": "ex:Velscope",
      "@type": "ex:Velscope",
      "skos:prefLabel": "Velscope",
      "ex:status": "normal"

    and another example: 

      "@id": "ex:Medical_History_Update",
      "@type": "ex:Medical_History_Update",
      "skos:prefLabel": "Medical History Update (HX)",
      "ex:medication": "Trazodone",
      "ex:medication_value": "50 mg",
      "ex:allergies": "No" 
      
      and the object properties should be in this format, for example:
      
      "@id": "ex:Patient_Record",
      "ex:has_medical_history": 
        "@id": "ex:Medical_History_Update"
      


{chat_history}
{context}
Question: {question}

"""
    prompt3 = PromptTemplate(
        template=template3, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )
    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        temperature=0,
        max_tokens=4000,
    )

    #print(f"Processing PDF: {pdf_path}")

    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()

    for idx, doc in enumerate(docs):


        headers_to_split_on = [
            ("Abstract", "Header 1"),
            ("Introduction", "Header 2"),
            ("Experimental Methods", "Header 3"),
            ("Results and Discussion", "Header 4"),
            ("Conclusion", "Header 5"),
            ("References", "Header 6")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        big_chunks_retriever.add_documents(md_header_splits)


    # Build QA interface for the PDF
    qa_interface1 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt1}
    )
     # Build QA interface for the PDF
    qa_interface2 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt2}
    )
    qa_interface3 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt3}
    )
    query1 = "Extract Velscope, Medical History Update (HX), Smoker, Dental Hygiene Diagnosis (DH), Chief Complaint (CC), Extra Oral/Intra Oral Exam (EOIO), Dentition, Sensitivity, Deposits, Bleeding, Perio Status, Oral Health Instructions (OHI),  Hygiene Treatment (HTX), Re-care Exam (RC), Next Visit (NV) from each document."
    result = qa_interface1({"question": query1})
    all_results1.append(result["answer"].strip('```json').strip('```').strip())

    queries2 = [
    "Extract the relationship between diagnostic tools and conditions as is_diagnosed_with. Avoid any extra explanation.",
    "Extract the relationship between diagnostic tools and oral health parameters as evaluates_parameter. Avoid any extra explanation.",
    "Extract the relationship between conditions and treatments as requires_treatment. Avoid any extra explanation.",
    "Extract the relationship between observations and dental properties as describes_property. Avoid any extra explanation.",
    "Extract the relationship between patient records and conditions as documents. Avoid any extra explanation.",
    "Extract the relationship between oral health parameters and dental properties as relates_to_property. Avoid any extra explanation.",
]

    for query in queries2:
        result2 = qa_interface2({"question": query})
        all_results2.append(result2["answer"].strip('```json').strip('```').strip())

    query3 = "Create a complete knowledge graph including all the nodes and the relationships extracted and stored in History. Use the URI defined for 'Classes' and 'Object properties' in ontology. Format the output in JSON-LD."
    result = qa_interface3({"question": query3})
    all_results.append(result["answer"].strip('```json').strip('```').strip())

    #print(f"Finished Processing PDF: {pdf_path}")
    msgs.clear()
    memory.clear()

    return all_results

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    try:
        results = process_pdf(tmp_path)
        for result in results:
            st.write(result)
    finally:
        os.remove(tmp_path)

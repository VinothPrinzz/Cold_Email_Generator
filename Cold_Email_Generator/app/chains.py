import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv() #find .env file and set it as environmental variable

os.getenv('GROQ_API_KEY') #get the api key from enviromental variable

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key= os.getenv('GROQ_API_KEY'),
            model_name= "llama-3.1-70b-versatile",
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            --> Scraped Text From Website:
            {pageData}
            --> Instruction:
            The scraped text is from the career's page of a website,
            Your job is to extract the job postings and return them in JSON format 
            containing following keys: 'role', 'experience', 'skills' and 'description'.
            Only return the valid JSON format(No Premable).
            """
        )

        chain_extract = prompt_extract |  self.llm
        response = chain_extract.invoke(input={'pageData': cleaned_text})

        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Context too big, Unable to parse jobs.")
        return response if isinstance(response, list) else [response]
    
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            --> Job Description:
            {jobDescription}
            --> Instruction:
            You are Vinoth, a business development executive at XYZ. XYZ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of XYZ
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase XYZ's portfolio: {linkList}
            Remember you are Vinoth, BDE at XYZ. 
            Do not provide a preamble.
            --> EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        response = chain_email.invoke({'jobDescription': str(job), 'linkList': links})
        return response.content
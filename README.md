# RAG-Chat

A Retrieval-Augmented Generation (RAG) system with a Streamlit-based chat interface, built using LangChain and OpenAI. The system is divided into two modules:

- `rag`: Contains the components for the RAG system.
- `chat`: Provides a user-friendly chat interface using Streamlit.

## How to Run

You have to have docker daemon running with docker-compose installed. I have tested this on a mac. 
There is currently a test not go throuh, it concerns the path parameters, which I need to fix. Just
delete the failing test, and it should run. Also, I have created a conda environment in pycharm and
am running the tests in this conda environment using python 3.10. You can probably just add a .venv virtual
environment and get the same effect. Usually CI/CD would have it's own environment for the testing. I 
didn't feel the need to create another docker container just for testing or creating a jenkins slave since
the purpose of this system was to show what I could do in a morning of work, and talk about the codebase
and the rag system in general. I thought this would be somehow the coding test.

1. Test, build, and run the containers using Docker Compose:
   ```bash
   chmod x+ ./run_tests_and_build.sh
   ./run_tests_and_build.sh
   ```

# Interesting Things to Know

There is a a shell script that I was using with chatgpt to feed in the code base. It's called
`print_code_base.sh`. Just fix the permissions and run it. You should get the output in the `code_base.md`
file in the `rag-chat` directory. I only tested for the `zsh` on mac os x. You can feed it a directory name
like `rag` or `chat` in order to focus on those particular folders. I have made some attempts to get rid
of some unwanted files.

Also, I have the `main_objective.md` file. This is a list of things to be done. The idea was to build an
agent to create the code base for me. This is definitely out of the scope of work for one morning's worth of work.
I have saved these files as relics, and was to continue to save other artifacts that I found relevant, for creating a bot.
It would be relatively expensive, and would require a number of different models. It is still, however, really neat.

Oh yes, I write this while trying to get away from the computer and back to vacation! The basic further algorithm would have
been to write a test and a piece of code, and then implement it in the main.py as part of the api. Then you test, build, and run
and then you would run the api call to perform the steps necessary to get the code up and running -- in an iterative fashion.

I want to thank Robert and Omair for giving me the inspiration to write this code. This was fun.g

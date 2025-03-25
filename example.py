from word_embedding import build_vector_database
from rag import RAGSystem

def main():
    # Step 1: Build the vector database from your text files
    # This only needs to be done once, or when you add new documents
    # build_vector_database(folder_path="data", output_dir="db")
    
    # Step 2: Initialize the RAG system
    # It will automatically load the pre-computed embeddings
    rag = RAGSystem()
    
    # Example project description/abstract
    my_project = """
        Robot-assisted bite acquisition systems need to ac-
    quire food items from a diverse range of food dishes that exist
    in the wild, in order to effectively empower users with mobility
    limitations who are unable to feed themselves. Existing state-of-
    the-art modular robotic pipelines for bite acquisition encounter
    various types of failure modes during deployment, ranging from
    perception to action-selection failures. While such systems have
    been designed to recover autonomously, autonomous recovery
    strategies can be sample-inefficient and difficult to develop
    for every modular component. We instead propose a human-
    in-the-loop failure recovery framework, GB-QUERY, that can
    choose to query a human for assistance. GB-QUERY models
    bite acquisition as a graph that includes autonomous actions,
    human query actions, and autonomous failure recovery, which
    corresponds to a contextual bandit for low-level action selection.
    We model the process of deciding which modules to query in
    the robotic pipeline as a graph search problem, where our
    framework relies on uncertainty estimates that come from each
    of the modules, along with estimates of the workload that each
    type of query imposes on the user. We show that our approach
    can recover from a wide range of failures, while minimizing user
    workload. We compare our approach to two heuristic failure
    recovery approaches, and show that our method outperforms
    these baselines, with minimal querying workload and number of
    queries required to recover from failures. We demonstrate this
    through experiments on offline food plate data and real robot
    experiments across a set of 4 in-the-wild food dishes.
    """
    
    # Step 3: Generate different versions of your abstract
    
    # General audience version
    general_version = rag.generate_storytelling_output(
        user_abstract=my_project,
        mode="general",
        k=3  # number of relevant documents to consider
    )
    print("\n=== General Version ===")
    print(general_version)
    
    # Investor pitch version
    investor_version = rag.generate_storytelling_output(
        user_abstract=my_project,
        mode="investor",
        k=3
    )
    print("\n=== Investor Pitch Version ===")
    print(investor_version)
    
    # Academic conference version
    conference_version = rag.generate_storytelling_output(
        user_abstract=my_project,
        mode="conference",
        k=3
    )
    print("\n=== Conference Abstract Version ===")
    print(conference_version)

if __name__ == "__main__":
    main() 
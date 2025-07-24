# main.py - Advanced Data Science Pipeline Execution

import os
import pandas as pd
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from tools import (
    NotebookCodeExecutor, 
    DataProfiler, 
    FeatureEngineer, 
    ModelValidator, 
    VisualizationGenerator
)
from agents import create_data_science_agents
from tasks import create_data_science_tasks


class AdvancedDataSciencePipeline:
    """
    Sophisticated Multi-Agent Data Science Pipeline
    
    This class orchestrates a team of specialized AI agents to conduct
    comprehensive machine learning projects with advanced analytics,
    intelligent feature engineering, and rigorous model validation.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", data_file_path: str = None):
        """
        Initialize the advanced data science pipeline.
        
        Args:
            api_key (str): OpenAI API key for LLM access
            model_name (str): LLM model to use for agents
            data_file_path (str): Path to the dataset CSV file
        """
        self.api_key = api_key
        self.model_name = model_name
        self.data_file_path = data_file_path
        
        # Setup environment and LLM
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.1)
        
        # Initialize shared namespace for tools
        self.shared_namespace = globals().copy()
        
        # Load data if provided
        if data_file_path:
            self.load_data(data_file_path)
        
        # Initialize agents and tasks
        self.agents = None
        self.tasks = None
        self.crew = None
        
        print("🚀 Advanced Data Science Pipeline Initialized")
        print(f"📊 Model: {model_name}")
        print(f"💾 Data: {data_file_path if data_file_path else 'Not loaded'}")
    
    
    def load_data(self, file_path: str, dataframe_name: str = "shared_df"):
        """
        Load dataset into the shared namespace.
        
        Args:
            file_path (str): Path to the CSV file
            dataframe_name (str): Name for the DataFrame in shared namespace
        """
        try:
            df = pd.read_csv(file_path)
            self.shared_namespace[dataframe_name] = df
            
            print(f"✅ Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"📋 Available as '{dataframe_name}' in shared namespace")
            
            # Display basic info
            print(f"🔍 Column names: {list(df.columns)}")
            print(f"📈 Data types: {df.dtypes.value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    
    def setup_agents_and_tasks(self, target_column: str = "Units Sold", dataframe_name: str = "shared_df"):
        """
        Initialize the specialized agent team and comprehensive task sequence.
        
        Args:
            target_column (str): Name of the target variable to predict
            dataframe_name (str): Name of the DataFrame in shared namespace
        """
        print("\n🤖 Setting up AI Agent Team...")
        
        # Create specialized agents
        self.agents = create_data_science_agents(self.llm, self.shared_namespace)
        
        print(f"✅ Created {len(self.agents)} specialized agents:")
        for role, agent in self.agents.items():
            print(f"   🎯 {agent.role}")
        
        # Create comprehensive task sequence
        print("\n📋 Designing task sequence...")
        self.tasks = create_data_science_tasks(
            self.agents, 
            target_column=target_column, 
            dataframe_name=dataframe_name
        )
        
        print(f"✅ Created {len(self.tasks)} sophisticated tasks:")
        for i, task in enumerate(self.tasks, 1):
            print(f"   {i}. {task.agent.role}")
        
        # Initialize the crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_rpm=10,
            memory=True
        )
        
        print("\n🚀 Advanced Data Science Crew Ready for Execution!")
    
    
    def execute_pipeline(self, target_column: str = "Units Sold", dataframe_name: str = "shared_df"):
        """
        Execute the complete data science pipeline with all agents.
        
        Args:
            target_column (str): Target variable for prediction
            dataframe_name (str): DataFrame name in shared namespace
            
        Returns:
            CrewAI execution result with comprehensive analysis
        """
        print("\n" + "="*80)
        print("🚀 LAUNCHING ADVANCED DATA SCIENCE PIPELINE")
        print("="*80)
        
        # Setup if not already done
        if not self.agents or not self.tasks:
            self.setup_agents_and_tasks(target_column, dataframe_name)
        
        # Validate data availability
        if dataframe_name not in self.shared_namespace:
            print(f"❌ Error: DataFrame '{dataframe_name}' not found in namespace")
            return None
        
        df = self.shared_namespace[dataframe_name]
        if target_column not in df.columns:
            print(f"❌ Error: Target column '{target_column}' not found in data")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f"📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"🎯 Target: {target_column}")
        print(f"🤖 Agents: {len(self.agents)} specialists")
        print(f"📋 Tasks: {len(self.tasks)} comprehensive steps")
        
        try:
            # Execute the crew
            print("\n🎬 Starting execution...")
            result = self.crew.kickoff()
            
            print("\n" + "="*80)
            print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*80)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Pipeline execution failed: {e}")
            return None
    
    
    def display_results(self, result):
        """
        Display the final results in a formatted manner.
        
        Args:
            result: CrewAI execution result
        """
        if result:
            print("\n" + "="*80)
            print("📊 FINAL RESULTS SUMMARY")
            print("="*80)
            
            try:
                display(Markdown(f"```\n{result.raw}\n```"))
            except:
                print(result.raw)
        else:
            print("❌ No results to display")
    
    
    def get_namespace_summary(self):
        """
        Display summary of variables available in the shared namespace.
        """
        print("\n🔍 SHARED NAMESPACE SUMMARY")
        print("-" * 50)
        
        dataframes = []
        models = []
        arrays = []
        other_vars = []
        
        for name, obj in self.shared_namespace.items():
            if name.startswith('_'):
                continue
                
            if isinstance(obj, pd.DataFrame):
                dataframes.append(f"📊 {name}: DataFrame {obj.shape}")
            elif hasattr(obj, 'predict') and hasattr(obj, 'fit'):
                models.append(f"🤖 {name}: {type(obj).__name__}")
            elif isinstance(obj, (list, tuple, pd.Series)) and len(str(obj)) < 100:
                arrays.append(f"📋 {name}: {type(obj).__name__} (length: {len(obj)})")
            elif not callable(obj) and not isinstance(obj, type):
                other_vars.append(f"🔧 {name}: {type(obj).__name__}")
        
        if dataframes:
            print("DataFrames:")
            for df in dataframes:
                print(f"  {df}")
        
        if models:
            print("\nModels:")
            for model in models:
                print(f"  {model}")
        
        if arrays:
            print("\nArrays/Lists:")
            for arr in arrays[:5]:  # Show first 5
                print(f"  {arr}")
            if len(arrays) > 5:
                print(f"  ... and {len(arrays) - 5} more")
        
        if other_vars:
            print(f"\nOther Variables: {len(other_vars)} items")


def main():
    """
    Main execution function - demonstrates the advanced pipeline.
    """
    print("🌟 ADVANCED MULTI-AGENT DATA SCIENCE PIPELINE")
    print("=" * 60)
    
    # Configuration
    API_KEY = ""  # Replace with your actual API key
    DATA_FILE = "/Users/saiaakash/Downloads/archive/House_Price_dataset.csv"  # Replace with your data file path
    TARGET_COLUMN = "price"  # Replace with your target column name
    
    try:
        # Initialize pipeline
        pipeline = AdvancedDataSciencePipeline(
            api_key=API_KEY,
            model_name="gpt-4o-mini",
            data_file_path=DATA_FILE
        )
        
        # Execute the complete pipeline
        results = pipeline.execute_pipeline(
            target_column=TARGET_COLUMN,
            dataframe_name="shared_df"
        )
        
        # Display results
        pipeline.display_results(results)
        
        # Show namespace summary
        pipeline.get_namespace_summary()
        
        print("\n🎉 Pipeline execution completed!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


# For Jupyter Notebook execution
def run_notebook_pipeline(api_key: str, data_file: str, target_column: str = "Units Sold"):
    """
    Convenience function for running the pipeline in Jupyter notebooks.
    
    Args:
        api_key (str): OpenAI API key
        data_file (str): Path to CSV data file
        target_column (str): Target variable name
    
    Returns:
        Tuple of (pipeline_instance, execution_results)
    """
    print("🚀 Initializing Advanced Data Science Pipeline for Notebook...")
    
    # Create pipeline instance
    pipeline = AdvancedDataSciencePipeline(
        api_key=api_key,
        model_name="gpt-4o-mini",
        data_file_path=data_file
    )
    
    # Execute pipeline
    results = pipeline.execute_pipeline(target_column=target_column)
    
    # Display results
    pipeline.display_results(results)
    
    return pipeline, results


# Example usage in Jupyter Notebook:
"""
# In a Jupyter cell:
import os
from main import run_notebook_pipeline

# Set your configuration
API_KEY = "sk-your-api-key-here"
DATA_FILE = "/content/your_dataset.csv"
TARGET_COLUMN = "Units Sold"

# Run the advanced pipeline
pipeline, results = run_notebook_pipeline(API_KEY, DATA_FILE, TARGET_COLUMN)

# Access the shared namespace for further analysis
pipeline.get_namespace_summary()
"""

if __name__ == "__main__":
    main()
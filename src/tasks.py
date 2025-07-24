# tasks.py - Comprehensive Task Definitions for Advanced Data Science Pipeline

from crewai import Task


def create_data_science_tasks(agents, target_column="Units Sold", dataframe_name="shared_df"):
    """
    Creates a sophisticated sequence of data science tasks that guide agents through
    a comprehensive machine learning pipeline with intelligent decision-making.
    """
    
    # Task 1: Strategic Planning and Project Architecture
    strategic_planning_task = Task(
        description=(
            f"As the Strategic Data Science Director, develop a comprehensive machine learning strategy "
            f"for predicting '{target_column}' using the available dataset '{dataframe_name}'. "
            f"Your strategic analysis must include:\n\n"
            f"🎯 **Project Objectives Analysis:**\n"
            f"- Define clear success metrics and performance benchmarks\n"
            f"- Identify potential business impact and model deployment considerations\n"
            f"- Assess data complexity and modeling challenges\n\n"
            f"📊 **Methodology Selection Strategy:**\n"
            f"- Recommend optimal modeling approaches based on problem characteristics\n"
            f"- Define validation strategy considering data temporal structure and size\n"
            f"- Specify feature engineering priorities and transformation strategies\n\n"
            f"🔧 **Technical Architecture Plan:**\n"
            f"- Design data preprocessing pipeline requirements\n"
            f"- Outline model selection and ensemble strategies\n"
            f"- Define performance evaluation and interpretation frameworks\n\n"
            f"⚠️ **Risk Assessment and Mitigation:**\n"
            f"- Identify potential data quality issues and modeling pitfalls\n"
            f"- Plan for overfitting prevention and generalization assurance\n"
            f"- Consider computational efficiency and scalability requirements\n\n"
            f"Your strategic plan should provide clear guidance for subsequent agents while "
            f"maintaining flexibility for data-driven adaptations during execution."
        ),
        expected_output=(
            "A comprehensive strategic document containing:\n"
            "1. **Executive Summary**: High-level project overview and success criteria\n"
            "2. **Detailed Methodology Plan**: Step-by-step approach with rationale\n"
            "3. **Technical Specifications**: Data processing and modeling requirements\n"
            "4. **Risk Mitigation Strategy**: Anticipated challenges and prevention measures\n"
            "5. **Success Metrics**: Quantitative benchmarks for model performance\n"
            "6. **Resource Requirements**: Computational and analytical resource planning\n\n"
            "The output should be structured, actionable, and serve as a roadmap for the entire team."
        ),
        agent=agents['strategy_director']
    )
    
    
    # Task 2: Comprehensive Data Intelligence Analysis
    data_analysis_task = Task(
        description=(
            f"As the Expert Data Intelligence Analyst, conduct an exhaustive analysis of the '{dataframe_name}' "
            f"dataset to provide actionable insights for the machine learning pipeline. Your analysis must "
            f"demonstrate advanced analytical thinking and provide strategic recommendations.\n\n"
            f"🔍 **Deep Data Exploration Requirements:**\n"
            f"- Execute comprehensive data profiling using the Intelligent Data Profiler tool\n"
            f"- Analyze data quality metrics, missing value patterns, and outlier distributions\n"
            f"- Investigate statistical properties and distribution characteristics of all variables\n"
            f"- Examine temporal patterns and seasonal trends if applicable\n\n"
            f"📈 **Advanced Statistical Analysis:**\n"
            f"- Perform correlation analysis and identify multicollinearity issues\n"
            f"- Conduct statistical significance tests for categorical-numerical relationships\n"
            f"- Analyze target variable distribution and identify potential transformation needs\n"
            f"- Investigate feature-target relationships using appropriate statistical measures\n\n"
            f"🎨 **Intelligent Visualization Strategy:**\n"
            f"- Generate comprehensive EDA visualizations using the Visualization Generator\n"
            f"- Create publication-ready plots that reveal key data insights\n"
            f"- Design visualizations that support data-driven decision making\n\n"
            f"💡 **Strategic Recommendations:**\n"
            f"- Provide specific data preprocessing recommendations based on findings\n"
            f"- Identify optimal feature engineering opportunities\n"
            f"- Flag potential modeling challenges and suggest mitigation strategies\n"
            f"- Recommend data collection improvements for future iterations"
        ),
        expected_output=(
            "A comprehensive data intelligence report including:\n"
            "1. **Data Quality Assessment**: Detailed analysis of completeness, consistency, and accuracy\n"
            "2. **Statistical Summary**: Advanced descriptive statistics and distribution analysis\n"
            "3. **Relationship Analysis**: Feature interactions and target variable correlations\n"
            "4. **Data Quality Issues**: Identified problems with severity assessment and solutions\n"
            "5. **Visualization Portfolio**: Publication-ready plots with analytical insights\n"
            "6. **Strategic Recommendations**: Actionable preprocessing and modeling guidance\n"
            "7. **Risk Factors**: Data-related risks that could impact model performance\n\n"
            "All findings must be supported by statistical evidence and visualizations."
        ),
        agent=agents['data_analyst'],
        context=[strategic_planning_task]
    )
    
    
    # Task 3: Advanced Feature Engineering and Optimization
    feature_engineering_task = Task(
        description=(
            f"As the Advanced Feature Engineering Specialist, design and implement a sophisticated "
            f"feature engineering pipeline that maximizes predictive power for '{target_column}' prediction. "
            f"Your approach must demonstrate advanced feature engineering expertise and domain understanding.\n\n"
            f"🧪 **Intelligent Feature Creation:**\n"
            f"- Use the Automated Feature Engineer tool to generate diverse feature transformations\n"
            f"- Apply domain-specific feature engineering techniques based on data characteristics\n"
            f"- Create polynomial features and interaction terms for potentially relevant combinations\n"
            f"- Implement advanced categorical encoding strategies (target encoding, frequency encoding)\n\n"
            f"📊 **Temporal and Numerical Transformations:**\n"
            f"- Engineer sophisticated temporal features if date/time variables are present\n"
            f"- Apply appropriate mathematical transformations to address skewness and scale issues\n"
            f"- Create binned versions of continuous variables to capture non-linear relationships\n"
            f"- Generate rolling statistics and lag features for time-dependent patterns\n\n"
            f"🎯 **Feature Selection and Optimization:**\n"
            f"- Implement statistical feature selection methods (correlation, mutual information)\n"
            f"- Apply regularization-based feature selection techniques\n"
            f"- Remove highly correlated features to address multicollinearity\n"
            f"- Validate feature importance using target-based statistical tests\n\n"
            f"✅ **Quality Assurance:**\n"
            f"- Ensure all engineered features maintain data integrity\n"
            f"- Validate feature distributions and handle edge cases appropriately\n"
            f"- Document feature engineering rationale and transformation logic\n"
            f"- Create robust preprocessing pipeline for production deployment"
        ),
        expected_output=(
            "A comprehensive feature engineering report containing:\n"
            "1. **Feature Engineering Strategy**: Detailed rationale for chosen transformations\n"
            "2. **Engineered Features Catalog**: Complete list with descriptions and statistical properties\n"
            "3. **Feature Selection Analysis**: Statistical justification for included/excluded features\n"
            "4. **Data Pipeline Code**: Production-ready preprocessing and transformation code\n"
            "5. **Feature Quality Metrics**: Validation results and quality assessments\n"
            "6. **Domain Insights**: Business-relevant interpretations of engineered features\n"
            "7. **Performance Impact Analysis**: Expected contribution to model performance\n\n"
            "All engineered features must be properly validated and ready for model training."
        ),
        agent=agents['feature_specialist'],
        context=[strategic_planning_task, data_analysis_task]
    )
    
    
    # Task 4: Advanced Model Architecture and Implementation
    model_development_task = Task(
        description=(
            f"As the Machine Learning Architecture Expert, design and implement sophisticated regression "
            f"models optimized for '{target_column}' prediction. Your implementation must demonstrate "
            f"advanced ML engineering skills and optimal model architecture decisions.\n\n"
            f"🏗️ **Model Architecture Design:**\n"
            f"- Implement multiple advanced regression algorithms (Random Forest, Gradient Boosting, XGBoost)\n"
            f"- Design ensemble strategies combining complementary model strengths\n"
            f"- Apply advanced hyperparameter optimization using grid search or Bayesian optimization\n"
            f"- Implement proper regularization techniques to prevent overfitting\n\n"
            f"⚙️ **Advanced Implementation Requirements:**\n"
            f"- Use sophisticated train/validation/test splitting strategies\n"
            f"- Implement cross-validation with appropriate stratification\n"
            f"- Apply feature scaling and normalization where algorithmically appropriate\n"
            f"- Handle class imbalance issues if present in the target variable\n\n"
            f"🎛️ **Model Optimization:**\n"
            f"- Perform systematic hyperparameter tuning with statistical validation\n"
            f"- Implement early stopping and regularization techniques\n"
            f"- Optimize for both performance and computational efficiency\n"
            f"- Create model persistence and loading mechanisms for deployment\n\n"
            f"📊 **Performance Evaluation:**\n"
            f"- Calculate comprehensive regression metrics (MAE, MSE, RMSE, R², MAPE)\n"
            f"- Perform residual analysis and assumption validation\n"
            f"- Assess model stability across different data segments\n"
            f"- Generate detailed performance benchmarking reports"
        ),
        expected_output=(
            "A comprehensive model development report including:\n"
            "1. **Model Architecture Documentation**: Detailed description of chosen algorithms and rationale\n"
            "2. **Implementation Code**: Production-ready model training and prediction code\n"
            "3. **Hyperparameter Optimization Results**: Systematic tuning process and optimal parameters\n"
            "4. **Performance Metrics**: Comprehensive evaluation across multiple metrics and validation sets\n"
            "5. **Model Comparison Analysis**: Detailed comparison of different algorithmic approaches\n"
            "6. **Computational Efficiency Analysis**: Training time, prediction speed, and resource usage\n"
            "7. **Deployment Readiness Assessment**: Model serialization and production deployment preparation\n\n"
            "All models must be fully trained, validated, and ready for comprehensive evaluation."
        ),
        agent=agents['ml_architect'],
        context=[strategic_planning_task, data_analysis_task, feature_engineering_task]
    )
    
    
    # Task 5: Rigorous Model Validation and Diagnostics
    model_validation_task = Task(
        description=(
            f"As the Model Validation and Performance Analyst, conduct comprehensive model validation "
            f"using advanced statistical techniques to ensure model reliability, generalizability, and "
            f"deployment readiness. Your analysis must meet the highest standards of statistical rigor.\n\n"
            f"🔬 **Advanced Validation Framework:**\n"
            f"- Execute comprehensive model validation using the Advanced Model Validator tool\n"
            f"- Perform k-fold cross-validation with statistical significance testing\n"
            f"- Implement time-series validation if temporal structure is present\n"
            f"- Conduct bootstrap validation for confidence interval estimation\n\n"
            f"📈 **Statistical Diagnostics:**\n"
            f"- Perform detailed residual analysis and assumption testing\n"
            f"- Test for homoscedasticity, normality, and independence of residuals\n"
            f"- Analyze prediction intervals and uncertainty quantification\n"
            f"- Conduct sensitivity analysis for feature importance stability\n\n"
            f"🎯 **Performance Robustness Assessment:**\n"
            f"- Evaluate model performance across different data segments\n"
            f"- Assess stability under various data perturbations\n"
            f"- Test generalization using holdout datasets and temporal validation\n"
            f"- Analyze feature importance consistency across validation folds\n\n"
            f"⚠️ **Risk Assessment and Quality Assurance:**\n"
            f"- Detect potential overfitting using learning curve analysis\n"
            f"- Identify data leakage and validate temporal integrity\n"
            f"- Assess model fairness and bias across different data segments\n"
            f"- Evaluate computational stability and numerical precision"
        ),
        expected_output=(
            "A comprehensive model validation report containing:\n"
            "1. **Validation Framework Summary**: Detailed methodology and statistical approaches used\n"
            "2. **Cross-Validation Results**: Comprehensive CV performance with confidence intervals\n"
            "3. **Statistical Diagnostics**: Residual analysis, assumption testing, and diagnostic plots\n"
            "4. **Robustness Analysis**: Performance stability across different conditions and data segments\n"
            "5. **Risk Assessment**: Identified risks, biases, and potential failure modes\n"
            "6. **Model Reliability Metrics**: Statistical measures of prediction reliability and uncertainty\n"
            "7. **Deployment Recommendations**: Specific guidance for production deployment and monitoring\n\n"
            "All validation results must include statistical significance testing and confidence intervals."
        ),
        agent=agents['validation_expert'],
        context=[strategic_planning_task, data_analysis_task, feature_engineering_task, model_development_task]
    )
    
    
    # Task 6: Executive Insights and Communication
    insights_communication_task = Task(
        description=(
            f"As the Data Visualization and Insights Communicator, synthesize all analytical findings "
            f"into compelling, actionable business intelligence that effectively communicates the complete "
            f"machine learning solution for '{target_column}' prediction to diverse stakeholder audiences.\n\n"
            f"📊 **Comprehensive Visualization Strategy:**\n"
            f"- Generate publication-ready model performance visualizations using the Visualization Generator\n"
            f"- Create executive dashboards summarizing key findings and model insights\n"
            f"- Design feature importance and interpretation visualizations\n"
            f"- Develop comparative analysis charts showing model performance across different approaches\n\n"
            f"💼 **Business Intelligence Synthesis:**\n"
            f"- Translate technical findings into clear business value propositions\n"
            f"- Identify actionable insights that drive strategic decision-making\n"
            f"- Quantify expected business impact and ROI from model deployment\n"
            f"- Provide specific recommendations for operational implementation\n\n"
            f"📋 **Stakeholder Communication:**\n"
            f"- Create executive summary suitable for C-level presentation\n"
            f"- Develop technical documentation for data science and engineering teams\n"
            f"- Design user-friendly model interpretation guides\n"
            f"- Provide clear guidance on model limitations and appropriate use cases\n\n"
            f"🎯 **Future Strategy and Recommendations:**\n"
            f"- Outline model monitoring and maintenance requirements\n"
            f"- Suggest data collection improvements for enhanced performance\n"
            f"- Recommend next steps for model iteration and enhancement\n"
            f"- Identify opportunities for model expansion and additional use cases"
        ),
        expected_output=(
            "A comprehensive business intelligence package including:\n"
            "1. **Executive Summary**: High-level overview with key findings and business impact\n"
            "2. **Visual Analytics Portfolio**: Publication-ready charts, graphs, and interactive dashboards\n"
            "3. **Model Performance Report**: Detailed analysis of model accuracy, reliability, and limitations\n"
            "4. **Feature Insights Analysis**: Business interpretation of key predictive factors\n"
            "5. **Implementation Roadmap**: Step-by-step deployment and operationalization plan\n"
            "6. **ROI Analysis**: Quantified business value and expected return on investment\n"
            "7. **Strategic Recommendations**: Future improvements and expansion opportunities\n\n"
            "All communications must be tailored for appropriate audiences with clear action items."
        ),
        agent=agents['insights_communicator'],
        context=[strategic_planning_task, data_analysis_task, feature_engineering_task, 
                model_development_task, model_validation_task]
    )
    
    
    return [
        strategic_planning_task,
        data_analysis_task,
        feature_engineering_task,
        model_development_task,
        model_validation_task,
        insights_communication_task
    ]
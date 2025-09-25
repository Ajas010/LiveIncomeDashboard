import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

st.set_page_config(
    page_title="Income Live Dashboard",
    page_icon='💸',
    layout='wide',
)

st.title('Live Income Data Monitoring App')


# Create sample data if the CSV file doesn't exist
@st.cache_data
def load_sample_data():
    # Sample income data structure
    np.random.seed(42)
    n_samples = 100

    occupations = ['Engineer', 'Doctor', 'Teacher', 'Manager', 'Sales', 'Researcher']
    marital_statuses = ['Never-married', 'Married', 'Divorced', 'Widowed', 'Separated']

    data = {
        'age': np.random.randint(20, 65, n_samples),
        'occupation': np.random.choice(occupations, n_samples),
        'marital-status': np.random.choice(marital_statuses, n_samples),
        'hours-per-week': np.random.randint(20, 60, n_samples),
        'income': np.random.choice(['<=50K', '>50K'], n_samples, p=[0.7, 0.3])
    }

    return pd.DataFrame(data)


# Try to load CSV, if not found use sample data
try:
    df = pd.read_csv('incomedata.csv')
    st.success("Loaded data from incomedata.csv")
except FileNotFoundError:
    st.warning("incomedata.csv not found. Using sample data instead.")
    df = load_sample_data()

# Filter
st.sidebar.header("Filters")
job_filter = st.sidebar.selectbox('Choose a job', df['occupation'].unique(), index=0)

# Add other filters
marital_filter = st.sidebar.multiselect(
    'Marital Status',
    options=df['marital-status'].unique(),
    default=df['marital-status'].unique()
)

age_range = st.sidebar.slider(
    'Age Range',
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(25, 55)
)

# Apply filters
filtered_df = df[
    (df['occupation'] == job_filter) &
    (df['marital-status'].isin(marital_filter)) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1])
    ]

st.sidebar.info(f"Showing {len(filtered_df)} records")

placeholder = st.empty()

# Add a stop button
if 'running' not in st.session_state:
    st.session_state.running = True

stop_button = st.sidebar.button('Stop Live Update')

if stop_button:
    st.session_state.running = False
    st.sidebar.success("Live update stopped")

start_button = st.sidebar.button('Start Live Update')

if start_button:
    st.session_state.running = True
    st.sidebar.success("Live update started")

# Duration selector
duration = st.sidebar.selectbox(
    'Update Duration (seconds)',
    options=[30, 60, 120, 300],
    index=2
)

if st.session_state.running:
    for i in range(duration):
        if not st.session_state.running:
            break

        # Create modified data for live updates
        temp_df = filtered_df.copy()
        temp_df['new_age'] = temp_df['age'] * np.random.choice(range(1, 5))
        temp_df['whpw_new'] = temp_df['hours-per-week'] * np.random.choice(range(1, 5))

        # Creating KPIs
        avg_age = np.mean(temp_df['new_age'])
        count_married = int(
            temp_df[temp_df['marital-status'] == 'Never-married']['marital-status'].count() + np.random.choice(
                range(1, 30)))
        hpw = np.mean(temp_df['whpw_new'])

        with placeholder.container():
            # Creating columns
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            # Filling metrics
            kpi1.metric(
                label='Average Modified Age',
                value=round(avg_age, 1),
                delta=round(avg_age - np.mean(temp_df['age']), 1)
            )
            kpi2.metric(
                label='Never Married Count',
                value=count_married,
                delta=count_married - len(temp_df[temp_df['marital-status'] == 'Never-married'])
            )
            kpi3.metric(
                label='Working Hours/W',
                value=round(hpw, 1),
                delta=round(hpw - np.mean(temp_df['hours-per-week']), 1)
            )
            kpi4.metric(
                label='Total Records',
                value=len(temp_df)
            )

            # Create columns for charts
            figCol1, figCol2 = st.columns(2)

            with figCol1:
                st.markdown('### Age vs Marital Status')
                if not temp_df.empty:
                    fig = px.density_heatmap(
                        data_frame=temp_df,
                        y='new_age',
                        x='marital-status',
                        title="Age Distribution by Marital Status"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for the selected filters")

            with figCol2:
                st.markdown('### Age Distribution')
                if not temp_df.empty:
                    fig2 = px.histogram(
                        data_frame=temp_df,
                        x='new_age',
                        nbins=20,
                        title="Age Distribution Histogram"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data available for the selected filters")

            # Additional chart
            st.markdown('### Marital Status Distribution')
            if not temp_df.empty:
                fig3 = px.pie(
                    data_frame=temp_df,
                    names='marital-status',
                    title="Marital Status Distribution"
                )
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown('### Data View')
            st.dataframe(temp_df.head(10), use_container_width=True)

            # Progress bar
            progress_bar = st.progress((i + 1) / duration)
            st.write(f"Update {i + 1}/{duration} - Next update in 1 second")

        time.sleep(1)
else:
    # Static view when live update is stopped
    with placeholder.container():
        st.info("Live update is paused. Click 'Start Live Update' to begin.")

        if not filtered_df.empty:
            # Static KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            kpi1.metric(label='Average Age', value=round(np.mean(filtered_df['age']), 1))
            kpi2.metric(label='Never Married Count',
                        value=len(filtered_df[filtered_df['marital-status'] == 'Never-married']))
            kpi3.metric(label='Avg Hours/Week', value=round(np.mean(filtered_df['hours-per-week']), 1))
            kpi4.metric(label='Total Records', value=len(filtered_df))

            # Static charts
            figCol1, figCol2 = st.columns(2)

            with figCol1:
                st.markdown('### Age vs Marital Status')
                fig = px.density_heatmap(
                    data_frame=filtered_df,
                    y='age',
                    x='marital-status'
                )
                st.plotly_chart(fig, use_container_width=True)

            with figCol2:
                st.markdown('### Age Distribution')
                fig2 = px.histogram(data_frame=filtered_df, x='age')
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('### Data View')
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No data available for the selected filters")

st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.info("""
1. Select filters from the sidebar
2. Click 'Start Live Update' to begin real-time monitoring
3. Use 'Stop Live Update' to pause
4. Adjust duration as needed
""")
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
import random
import tempfile
import os
from fingraph.data.data_engine import TransactionLoader

# Page Config
st.set_page_config(
    page_title="FinGraph AML Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FinGraph: AI-Powered AML Detection")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Live Monitor", "🔍 Forensic Tool", "🕸️ Network Graph"])

# --- Tab 1: Live Monitor ---
with tab1:
    st.header("Real-time System Status")
    
    # Simulate real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Auto-refresh simulation
    placeholder = st.empty()
    
    # Just show a static snapshot for the MVP to avoid infinite loops in dev
    # In production, use st.empty() and a loop
    
    requests_sec = random.randint(120, 150)
    avg_latency = random.randint(40, 60)
    fraud_detected = random.randint(2, 8)
    active_nodes = 15420

    col1.metric("Requests / Sec", f"{requests_sec}", "12%")
    col2.metric("Avg Latency (ms)", f"{avg_latency}ms", "-5%")
    col3.metric("Fraud Alerts (1h)", f"{fraud_detected}", "2", delta_color="inverse")
    col4.metric("Active Graph Nodes", f"{active_nodes}", "150")

    st.subheader("Recent Alerts")
    alert_data = pd.DataFrame({
        "Timestamp": [pd.Timestamp.now() - pd.Timedelta(minutes=i*5) for i in range(5)],
        "Transaction ID": [f"TXN-{random.randint(1000,9999)}" for _ in range(5)],
        "Risk Score": [0.95, 0.88, 0.92, 0.76, 0.81],
        "Type": ["Smurfing", "Cycle", "Smurfing", "High Value", "Cycle"]
    })
    st.dataframe(alert_data, use_container_width=True)

# --- Tab 2: Forensic Tool ---
with tab2:
    st.header("Smurfing Detection Engine")
    st.markdown("Upload transaction logs (CSV) to detect structuring patterns.")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Ingesting data into DuckDB..."):
            # Save to temp file for DuckDB to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = TransactionLoader(tmp_path)
                suspects = loader.get_structuring_suspects()
                
                st.success(f"Analysis Complete. Found {len(suspects)} suspicious accounts.")
                
                if not suspects.empty:
                    st.dataframe(suspects, use_container_width=True)
                    
                    # Download report
                    csv = suspects.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Suspicious Activity Report (SAR)",
                        csv,
                        "sar_report.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.info("No structuring patterns detected in this batch.")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                os.unlink(tmp_path)

# --- Tab 3: Network Graph ---
with tab3:
    st.header("GNN Risk Analysis & Visualization")
    
    from fingraph.core.inference import AMLInference
    
    # Initialize Inference Engine (Lazy load)
    if 'inference_engine' not in st.session_state:
        try:
            st.session_state.inference_engine = AMLInference()
            # Pre-load data if possible, or load on demand
            # st.session_state.inference_engine.load_data() 
        except Exception as e:
            st.error(f"Could not initialize inference engine: {e}")
    
    # --- Demo Data Section ---
    with st.expander("ℹ️ Need an Account ID? (Demo Data)"):
        st.write("Try these accounts to see the model in action:")
        if st.button("Load Sample Accounts"):
            engine = st.session_state.inference_engine
            if engine.data is None:
                with st.spinner("Loading data..."):
                    engine.load_data()
            
            conn = engine.loader.conn
            try:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🚨 Known Launderers")
                    bad_actors = conn.execute("SELECT DISTINCT From_Account FROM transactions WHERE Is_Laundering = 1 LIMIT 5").fetchdf()
                    st.dataframe(bad_actors, hide_index=True)
                with c2:
                    st.markdown("### ✅ Normal Accounts")
                    good_actors = conn.execute("SELECT DISTINCT From_Account FROM transactions WHERE Is_Laundering = 0 LIMIT 5").fetchdf()
                    st.dataframe(good_actors, hide_index=True)
            except Exception as e:
                st.error(f"Could not load demo data: {e}")

    col_search, col_actions = st.columns([3, 1])
    
    with col_search:
        # Add a selectbox for quick access to high risk accounts
        if 'high_risk_accounts' not in st.session_state:
             st.session_state.high_risk_accounts = []
             
        target_account_input = st.text_input("Enter Account Number to Analyze", placeholder="e.g. 1000531")
        
    with col_actions:
        st.write("") # Spacer
        st.write("")
        analyze_btn = st.button("Analyze Risk", type="primary")
        scan_btn = st.button("Scan for High Risk", type="secondary")

    if scan_btn:
        engine = st.session_state.inference_engine
        with st.spinner("Scanning entire graph for high risk nodes..."):
            try:
                if engine.data is None:
                    engine.load_data()
                    engine.load_model()
                
                risky = engine.get_high_risk_accounts(limit=10)
                st.session_state.high_risk_accounts = [r['account'] for r in risky]
                st.success(f"Found {len(risky)} high risk accounts!")
                st.dataframe(risky)
            except Exception as e:
                st.error(f"Scan failed: {e}")

    # Use selected account from scan if available and input is empty
    target_account = target_account_input
    
    if analyze_btn and target_account:
        engine = st.session_state.inference_engine
        
        with st.spinner(f"Analyzing Account {target_account}..."):
            try:
                # Ensure data is loaded
                if engine.data is None:
                    engine.load_data()
                    engine.load_model()
                
                # 1. Get Prediction
                result = engine.predict_account(target_account)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    risk_score = result['risk_score']
                    is_laundering = result['is_laundering']
                    
                    # Display Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Score", f"{risk_score:.4f}")
                    m2.metric("Prediction", "LAUNDERING" if is_laundering else "Normal", 
                             delta="-High Risk" if is_laundering else "Safe",
                             delta_color="inverse")
                    
                    # 2. Visualize Subgraph (Ego Graph)
                    st.subheader("Transaction Neighborhood")
                    
                    # Get neighbors from the PyG data object or DuckDB
                    # Using DuckDB is easier for getting edge details
                    query = f"""
                    SELECT "From_Account", "To_Account", "Amount"
                    FROM transactions 
                    WHERE "From_Account" = '{target_account}' 
                       OR "To_Account" = '{target_account}'
                    LIMIT 50
                    """
                    edges_df = engine.loader.conn.execute(query).fetchdf()
                    
                    if edges_df.empty:
                        st.warning("No transactions found for this account.")
                    else:
                        # Build Graph
                        G = nx.DiGraph()
                        for _, row in edges_df.iterrows():
                            G.add_edge(row['From_Account'], row['To_Account'], amount=row['Amount'])
                            
                        # Layout
                        pos = nx.spring_layout(G, seed=42)
                        
                        # Edges
                        edge_x, edge_y = [], []
                        edge_text = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            edge_text.append(f"${edge[2]['amount']:.2f}")

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='text',
                            text=edge_text,
                            mode='lines')

                        # Nodes
                        node_x, node_y = [], []
                        node_text = []
                        node_colors = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                            if node == target_account:
                                node_colors.append('red')
                            else:
                                node_colors.append('blue')

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="top center",
                            hoverinfo='text',
                            marker=dict(
                                color=node_colors,
                                size=20,
                                line_width=2))

                        fig = go.Figure(data=[edge_trace, node_trace],
                                     layout=go.Layout(
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Analysis failed: {e}")

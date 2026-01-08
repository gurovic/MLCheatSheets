#!/usr/bin/env python3
"""
Add matplotlib illustrations to GNN cheatsheet HTML files.
"""

import re
from generate_gnn_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_gnn_basics(html_content, illustrations):
    """Add illustrations to GNN basics cheatsheet."""
    
    # Add graph structure after the first table
    graph_pattern = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 2\. Представление графа)'
    graph_img = create_img_tag(illustrations['gnn_graph_structure'], 
                               'Примеры графовых структур', '95%')
    html_content = re.sub(graph_pattern, 
                         r'</table>\n\n' + graph_img + r'  </div>\n<div class="block">\n    <h2>🔷 2. Представление графа', 
                         html_content, count=1)
    
    # Add message passing after the message passing code
    mp_pattern = r'(</ol>\s*</div>\s*<div class="block">\s*<h2>🔷 4\. GCN Layer)'
    mp_img = create_img_tag(illustrations['gnn_message_passing'], 
                           'Message Passing механизм', '95%')
    html_content = re.sub(mp_pattern, 
                         r'</ol>\n  </div>\n\n' + mp_img + r'\n  <div class="block">\n    <h2>🔷 4. GCN Layer', 
                         html_content, count=1)
    
    # Add aggregation functions after training code
    agg_pattern = r'(optimizer\.step\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 8\. Датасеты)'
    agg_img = create_img_tag(illustrations['gnn_aggregation'], 
                            'Функции агрегации в GNN', '95%')
    html_content = re.sub(agg_pattern, 
                         r'optimizer.step()</code></pre>\n  </div>\n\n' + agg_img + r'\n  <div class="block">\n    <h2>🔷 8. Датасеты', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_gcn(html_content, illustrations):
    """Add illustrations to GCN cheatsheet."""
    
    # Add GCN layer operation after the math formulas
    gcn_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Intuition)'
    gcn_img = create_img_tag(illustrations['gcn_layer_operation'], 
                            'GCN: Нормализация по степеням узлов', '95%')
    html_content = re.sub(gcn_pattern, 
                         r'</ul>\n  </div>\n\n' + gcn_img + r'\n  <div class="block">\n    <h2>🔷 3. Intuition', 
                         html_content, count=1)
    
    # Add normalization comparison after the training code
    norm_pattern = r"(print\(f'Epoch \{epoch\}, Loss: \{loss:.4f\}, Acc: \{acc:.4f\}'\)</code></pre>\s*</div>\s*<div class=\"block\">\s*<h2>🔷 8\. Задачи на графах)"
    norm_img = create_img_tag(illustrations['gcn_normalization'], 
                             'Зачем нужна нормализация в GCN', '95%')
    html_content = re.sub(norm_pattern, 
                         r"print(f'Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}')</code></pre>\n  </div>\n\n" + norm_img + r'\n  <div class="block">\n    <h2>🔷 8. Задачи на графах', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_gat(html_content, illustrations):
    """Add illustrations to GAT cheatsheet."""
    
    # Add attention mechanism after the math formulas
    att_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Intuition)'
    att_img = create_img_tag(illustrations['gat_attention'], 
                            'GAT Attention Mechanism', '95%')
    html_content = re.sub(att_pattern, 
                         r'</ul>\n  </div>\n\n' + att_img + r'\n  <div class="block">\n    <h2>🔷 3. Intuition', 
                         html_content, count=1)
    
    # Add multi-head attention after multi-head code
    multihead_pattern = r'(return F\.log_softmax\(x, dim=1\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 6\. PyTorch Geometric)'
    multihead_img = create_img_tag(illustrations['gat_multihead'], 
                                  'Multi-head Attention', '95%')
    html_content = re.sub(multihead_pattern, 
                         r'return F.log_softmax(x, dim=1)</code></pre>\n  </div>\n\n' + multihead_img + r'\n  <div class="block">\n    <h2>🔷 6. PyTorch Geometric', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_mpnn(html_content, illustrations):
    """Add illustrations to Message Passing Networks cheatsheet."""
    
    # Add MPNN framework after the math formulas
    mpnn_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Компоненты MPNN)'
    mpnn_img = create_img_tag(illustrations['mpnn_framework'], 
                             'Message Passing Neural Network Framework', '95%')
    html_content = re.sub(mpnn_pattern, 
                         r'</ul>\n  </div>\n\n' + mpnn_img + r'\n  <div class="block">\n    <h2>🔷 3. Компоненты MPNN', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_graph_embeddings(html_content, illustrations):
    """Add illustrations to Graph Embeddings cheatsheet."""
    
    # Add node embeddings visualization after methods table
    node_pattern = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. DeepWalk - основа)'
    node_img = create_img_tag(illustrations['embeddings_node'], 
                             'Node Embeddings: граф → векторное пространство', '95%')
    html_content = re.sub(node_pattern, 
                         r'</table>\n  </div>\n\n' + node_img + r'\n\n  <div class="block">\n    <h2>🔷 3. DeepWalk - основа', 
                         html_content, count=1)
    
    # Add graph classification pipeline after applications section
    graph_pattern = r'(model\.fit\(X_edges, y_edges\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 9\. Визуализация эмбеддингов)'
    graph_img = create_img_tag(illustrations['embeddings_graph_classification'], 
                               'Graph Classification Pipeline', '95%')
    html_content = re.sub(graph_pattern, 
                         r'model.fit(X_edges, y_edges)</code></pre>\n  </div>\n\n' + graph_img + r'\n\n  <div class="block">\n    <h2>🔷 9. Визуализация эмбеддингов', 
                         html_content, count=1)
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add illustrations
        modified_content = add_illustrations_func(html_content, illustrations)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✓ Successfully updated {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to add illustrations to all GNN cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to GNN cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/gnn_basics_cheatsheet.html', add_illustrations_to_gnn_basics),
        ('cheatsheets/gcn_cheatsheet.html', add_illustrations_to_gcn),
        ('cheatsheets/gat_cheatsheet.html', add_illustrations_to_gat),
        ('cheatsheets/message_passing_networks_cheatsheet.html', add_illustrations_to_mpnn),
        ('cheatsheets/graph_embeddings_cheatsheet.html', add_illustrations_to_graph_embeddings),
    ]
    
    success_count = 0
    for filepath, add_func in files_to_process:
        if process_html_file(filepath, add_func, illustrations):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(files_to_process)} files successfully updated")
    print("=" * 70)

if __name__ == '__main__':
    main()

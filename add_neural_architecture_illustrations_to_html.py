#!/usr/bin/env python3
"""
Add matplotlib illustrations to Neural Architecture cheatsheet HTML files.
"""

import re
from generate_neural_architecture_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="90%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 15px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_cnn_architectures(html_content, illustrations):
    """Add illustrations to CNN Architectures cheatsheet."""
    
    # Add LeNet architecture after LeNet section
    lenet_pattern = r'(<div class="block"><h2>🔷 1\. LeNet-5[^<]+</h2>.*?</div>)'
    lenet_img = create_img_tag(illustrations['lenet_architecture'], 
                              'LeNet-5 Architecture', '90%')
    html_content = re.sub(lenet_pattern, 
                         r'\1\n' + lenet_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    # Add VGG block after VGG section
    vgg_block_pattern = r'(<div class="block"><h2>🔷 3\. VGG[^<]+</h2>.*?</div>)'
    vgg_block_img = create_img_tag(illustrations['vgg_block'], 
                                   'VGG Building Block', '80%')
    html_content = re.sub(vgg_block_pattern, 
                         r'\1\n' + vgg_block_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    # Add AlexNet vs VGG comparison after comparison section
    vgg_pattern = r'(<div class="block"><h2>🔷 4\. Сравнение архитектур</h2>.*?</div>)'
    vgg_img = create_img_tag(illustrations['alexnet_vs_vgg'], 
                            'AlexNet vs VGG: Эволюция CNN архитектур', '95%')
    html_content = re.sub(vgg_pattern, 
                         r'\1\n' + vgg_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_resnet(html_content, illustrations):
    """Add illustrations to ResNet cheatsheet."""
    
    # Add residual block after the residual block description
    residual_pattern = r'(Вместо обучения H\(x\), обучаем остаток F\(x\) = H\(x\) - x</p>\s*</div>\s*<div class="block"><h2>🔷 3\.)'
    residual_img = create_img_tag(illustrations['residual_block'], 
                                  'ResNet Residual Block', '80%')
    html_content = re.sub(residual_pattern, 
                         r'Вместо обучения H(x), обучаем остаток F(x) = H(x) - x</p>\n  </div>\n\n' + residual_img + r'\n  <div class="block"><h2>🔷 3.', 
                         html_content, count=1, flags=re.DOTALL)
    
    # Add degradation problem illustration after variants table
    degradation_pattern = r'(</table>\s*</div>\s*<div class="block"><h2>🔷 5\. Bottleneck)'
    degradation_img = create_img_tag(illustrations['resnet_degradation'], 
                                     'ResNet: Решение проблемы деградации', '95%')
    html_content = re.sub(degradation_pattern, 
                         r'</table>\n  </div>\n\n' + degradation_img + r'\n  <div class="block"><h2>🔷 5. Bottleneck', 
                         html_content, count=1, flags=re.DOTALL)
    
    # Add skip connections illustration
    skip_pattern = r'(</ul>\s*</div>\s*<div class="block"><h2>🔷 8\. Лучшие практики)'
    skip_img = create_img_tag(illustrations['skip_connections'], 
                             'Типы Skip Connections', '95%')
    html_content = re.sub(skip_pattern, 
                         r'</ul>\n  </div>\n\n' + skip_img + r'\n  <div class="block"><h2>🔷 8. Лучшие практики', 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_inception_efficientnet(html_content, illustrations):
    """Add illustrations to Inception/EfficientNet cheatsheet."""
    
    # Add Inception module after Inception description
    inception_pattern = r'(<div class="block"><h2>🔷 1\. Суть Inception</h2>.*?</div>)'
    inception_img = create_img_tag(illustrations['inception_module'], 
                                   'Inception Module', '90%')
    html_content = re.sub(inception_pattern, 
                         r'\1\n' + inception_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    # Add EfficientNet scaling after EfficientNet description
    efficient_pattern = r'(<div class="block"><h2>🔷 5\. Суть EfficientNet</h2>.*?</div>)'
    efficient_img = create_img_tag(illustrations['efficientnet_scaling'], 
                                   'EfficientNet: Compound Scaling', '95%')
    html_content = re.sub(efficient_pattern, 
                         r'\1\n' + efficient_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_vision_transformers(html_content, illustrations):
    """Add illustrations to Vision Transformers cheatsheet."""
    
    # Add ViT architecture after the main idea
    vit_pattern = r'(</ul>\s*</div>\s*<div class="block"><h2>🔷 2\. Архитектура ViT)'
    vit_img = create_img_tag(illustrations['vision_transformer'], 
                            'Vision Transformer (ViT)', '95%')
    html_content = re.sub(vit_pattern, 
                         r'</ul>\n  </div>\n\n' + vit_img + r'\n  <div class="block"><h2>🔷 2. Архитектура ViT', 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_vision_transformers(html_content, illustrations):
    """Add illustrations to Vision Transformers cheatsheet."""
    
    # Add ViT architecture after the main idea
    vit_pattern = r'(<div class="block"><h2>🔷 1\. Что такое ViT</h2>.*?</div>)'
    vit_img = create_img_tag(illustrations['vision_transformer'], 
                            'Vision Transformer (ViT)', '95%')
    html_content = re.sub(vit_pattern, 
                         r'\1\n' + vit_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_transformers(html_content, illustrations):
    """Add illustrations to Transformers cheatsheet."""
    
    # Add transformer architecture after main idea
    transformer_pattern = r'(<div class="block"><h2>🔷 1\. Суть</h2>.*?</div>)'
    transformer_img = create_img_tag(illustrations['transformer_architecture'], 
                                    'Transformer Architecture', '95%')
    html_content = re.sub(transformer_pattern, 
                         r'\1\n' + transformer_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_attention_mechanism(html_content, illustrations):
    """Add illustrations to Attention Mechanism cheatsheet."""
    
    # Add attention mechanism visualization
    attention_pattern = r'(<div class="block"><h2>🔷 1\. Суть Attention</h2>.*?</div>)'
    attention_img = create_img_tag(illustrations['attention_mechanism'], 
                                   'Attention Mechanism', '90%')
    html_content = re.sub(attention_pattern, 
                         r'\1\n' + attention_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_multi_head_attention(html_content, illustrations):
    """Add illustrations to Multi-Head Attention cheatsheet."""
    
    # Add multi-head attention visualization
    mha_pattern = r'(<div class="block"><h2>🔷 1\. Что такое Multi-head Attention</h2>.*?</div>)'
    mha_img = create_img_tag(illustrations['multi_head_attention'], 
                            'Multi-Head Attention', '95%')
    html_content = re.sub(mha_pattern, 
                         r'\1\n' + mha_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_autoencoders(html_content, illustrations):
    """Add illustrations to Autoencoders cheatsheet."""
    
    # Add autoencoder architecture
    ae_pattern = r'(<div class="block"><h2>🔷 1\. Суть</h2>.*?</div>)'
    ae_img = create_img_tag(illustrations['autoencoder_architecture'], 
                           'Autoencoder Architecture', '95%')
    html_content = re.sub(ae_pattern, 
                         r'\1\n' + ae_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_vae(html_content, illustrations):
    """Add illustrations to VAE cheatsheet."""
    
    # Add VAE architecture with reparameterization trick
    vae_pattern = r'(<div class="block"><h2>🔷 1\. Суть VAE</h2>.*?</div>)'
    vae_img = create_img_tag(illustrations['vae_architecture'], 
                            'Variational Autoencoder (VAE)', '95%')
    html_content = re.sub(vae_pattern, 
                         r'\1\n' + vae_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_gan(html_content, illustrations):
    """Add illustrations to GAN cheatsheet."""
    
    # Add GAN architecture
    gan_pattern = r'(<div class="block"><h2>🔷 1\. Суть GAN</h2>.*?</div>)'
    gan_img = create_img_tag(illustrations['gan_architecture'], 
                            'Generative Adversarial Network (GAN)', '95%')
    html_content = re.sub(gan_pattern, 
                         r'\1\n' + gan_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_diffusion_models(html_content, illustrations):
    """Add illustrations to Diffusion Models cheatsheet."""
    
    # Add diffusion process visualization
    diffusion_pattern = r'(<div class="block"><h2>🔷 1\. Суть диффузии</h2>.*?</div>)'
    diffusion_img = create_img_tag(illustrations['diffusion_process'], 
                                   'Diffusion Forward & Reverse Process', '95%')
    html_content = re.sub(diffusion_pattern, 
                         r'\1\n' + diffusion_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_capsule_networks(html_content, illustrations):
    """Add illustrations to Capsule Networks cheatsheet."""
    
    # Add capsule network architecture
    capsule_pattern = r'(<div class="block"><h2>1\. Мотивация</h2>.*?</div>)'
    capsule_img = create_img_tag(illustrations['capsule_network'], 
                                 'Capsule Network Architecture', '90%')
    html_content = re.sub(capsule_pattern, 
                         r'\1\n' + capsule_img, 
                         html_content, count=1, flags=re.DOTALL)
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add illustrations
        modified_content = add_illustrations_func(html_content, illustrations)
        
        # Only write if content changed
        if modified_content != html_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  ✓ Successfully updated {filepath}")
            return True
        else:
            print(f"  ℹ No changes needed for {filepath}")
            return False
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False

def main():
    """Main function to add illustrations to all Neural Architecture cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to Neural Architecture cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/cnn_architectures_cheatsheet.html', add_illustrations_to_cnn_architectures),
        ('cheatsheets/resnet_cheatsheet.html', add_illustrations_to_resnet),
        ('cheatsheets/inception_efficientnet_cheatsheet.html', add_illustrations_to_inception_efficientnet),
        ('cheatsheets/vision_transformers_vit_cheatsheet.html', add_illustrations_to_vision_transformers),
        ('cheatsheets/transformers_cheatsheet.html', add_illustrations_to_transformers),
        ('cheatsheets/attention_mechanism_cheatsheet.html', add_illustrations_to_attention_mechanism),
        ('cheatsheets/multi_head_attention_cheatsheet.html', add_illustrations_to_multi_head_attention),
        ('cheatsheets/autoencoders_cheatsheet.html', add_illustrations_to_autoencoders),
        ('cheatsheets/vae_cheatsheet.html', add_illustrations_to_vae),
        ('cheatsheets/gan_cheatsheet.html', add_illustrations_to_gan),
        ('cheatsheets/diffusion_models_cheatsheet.html', add_illustrations_to_diffusion_models),
        ('cheatsheets/capsule_networks_cheatsheet.html', add_illustrations_to_capsule_networks),
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

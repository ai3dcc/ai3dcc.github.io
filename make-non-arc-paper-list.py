#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from collections import namedtuple
import sys


def _print(args, text, **kwargs):
    if not args.quiet:
        print(text, **kwargs)


class Paper(namedtuple("Paper", [
        "title",
        "url",
        "authors",
        "links"
    ])):
    pass


class Conference(namedtuple("Conference", ["name"])):
    pass


class Link(namedtuple("Link", ["name", "url", "html", "text"])):
    pass


def author_list(authors):
    return authors.split(",")


publications = [

    Paper(
        "Ray Conditioning: Trading Photo-consistency for Photo-realism in Multi-view Image Generation",
        "https://ray-cond.github.io/",
        author_list("Eric M Chen; Sidhanth Holalkere; Ruyu Yan; Kai Zhang; Abe Davis"),
        [   Link("Abstract", None, "Multi-view image generation attracts particular attention these days due to its promising 3D-related applications, e.g., image viewpoint editing. Most existing methods follow a paradigm where a 3D representation is first synthesized, and then rendered into 2D images to ensure photo-consistency across viewpoints. However, such explicit bias for photo-consistency sacrifices photo-realism, causing geometry artifacts and loss of fine-scale details when these methods are applied to edit real images. To address this issue, we propose ray conditioning, a geometry-free alternative that relaxes the photo-consistency constraint. Our method generates multi-view images by conditioning a 2D GAN on a light field prior. With explicit viewpoint control, state-of-the-art photo-realism and identity consistency, our method is particularly suited for the viewpoint editing task.", None),
            Link("Paper", "https://arxiv.org/pdf/2304.13681.pdf", None, None),
        ]
    ),

    Paper(
        "Autodecoding Latent 3D Diffusion Models",
        "https://snap-research.github.io/3DVADER/",
        author_list("Evangelos Ntavelis; Aliaksandr Siarohin; Kyle B Olszewski; Chaoyang Wang; Luc Van Gool; Sergey Tulyakov"),
        [   Link("Abstract", None, "We present a novel approach to the generation of static and articulated 3D assets that has a 3D autodecoder at its core. The 3D autodecoder framework embeds properties learned from the target dataset in the latent space, which can then be decoded into a volumetric representation for rendering view-consistent appearance and geometry. We then identify the appropriate intermediate volumetric latent space, and introduce robust normalization and de-normalization operations to learn a 3D diffusion from 2D images or monocular videos of rigid or articulated objects. Our approach is flexible enough to use either existing camera supervision or no camera information at all -- instead efficiently learning it during training. Our evaluations demonstrate that our generation results outperform state-of-the-art alternatives on various benchmark datasets and metrics, including multi-view image datasets of synthetic objects, real in-the-wild videos of moving people, and a large-scale, real video dataset of static objects.", None),
            Link("Paper", "https://arxiv.org/pdf/2307.05445.pdf", None, None),
        ]
    ),

    Paper(
        "Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models",
        "https://lukashoel.github.io/text-to-room/",
        author_list("Lukas Höllein; Ang Cao; Andrew Owens; Justin Johnson; Matthias Niessner"),
        [   Link("Abstract", None, "We present Text2Room, a method for generating room-scale textured 3D meshes from a given text prompt as input. To this end, we leverage pre-trained 2D text-to-image models to synthesize a sequence of images from different poses. In order to lift these outputs into a consistent 3D scene representation, we combine monocular depth estimation with a text-conditioned inpainting model. The core idea of our approach is a tailored viewpoint selection such that the content of each image can be fused into a seamless, textured 3D mesh. More specifically, we propose a continuous alignment strategy that iteratively fuses scene frames with the existing geometry to create a seamless mesh. Unlike existing works that focus on generating single objects [56, 41] or zoom-out trajectories [18] from text, our method generates complete 3D scenes with multiple objects and explicit 3D geometry. We evaluate our approach using qualitative and quantitative metrics, demonstrating it as the first method to generate room-scale 3D geometry with compelling textures from only text as input.", None),
            Link("Paper", "https://arxiv.org/pdf/2303.11989.pdf", None, None),
            Link("Poster", "posters/iccv23_poster_text2room.pdf", None, None),
        ]
    ),

    Paper(
        "BallGAN: 3D-aware Image Synthesis with a Spherical Background",
        "https://minjung-s.github.io/ballgan",
        author_list("Minjung Shin; Yunji Seo; Jeongmin Bae; Young Sun Choi; Hyunsu Kim; Hyeran Byun; Youngjung Uh"),
        [   Link("Abstract", None, "3D-aware GANs aim to synthesize realistic 3D scenes that can be rendered in arbitrary camera viewpoints, generating high-quality images with well-defined geometry. As 3D content creation becomes more popular, the ability to generate foreground objects separately from the background has become a crucial property. Existing methods have been developed regarding overall image quality, but they can not generate foreground objects only and often show degraded 3D geometry. In this work, we propose to represent the background as a spherical surface for multiple reasons inspired by computer graphics. Our method naturally provides foreground-only 3D synthesis facilitating easier 3D content creation. Furthermore, it improves the foreground geometry of 3D-aware GANs and the training stability on datasets with complex backgrounds.", None),
            Link("Paper", "https://arxiv.org/pdf/2301.09091.pdf", None, None),
            Link("Poster", "posters/ballgan_workshop.pdf", None, None),
        ]
    ),

    Paper(
        "3D-aware Blending with Generative NeRFs",
        "https://blandocs.github.io/blendnerf",
        author_list("Hyunsu Kim; Gayoung Lee; Yunjey Choi; Jin-Hwa Kim; Jun-Yan Zhu"),
        [   Link("Abstract", None, "Image blending aims to combine multiple images seamlessly. It remains challenging for existing 2D-based methods, especially when input images are misaligned due to differences in 3D camera poses and object shapes. To tackle these issues, we propose a 3D-aware blending method using generative Neural Radiance Fields (NeRF), including two key components: 3D-aware alignment and 3D-aware blending. For 3D-aware alignment, we first estimate the camera pose of the reference image with respect to generative NeRFs and then perform pose alignment for objects. To further leverage 3D information of the generative NeRF, we propose 3D-aware blending that utilizes volume density and blends on the NeRF's latent space, rather than raw pixel space.", None),
            Link("Paper", "https://arxiv.org/pdf/2302.06608.pdf", None, None),
            Link("Poster", "posters/blendnerf_poster_workshop.pdf", None, None),
        ]
    ),

    Paper(
        "Scalable 3D Captioning with Pretrained Models",
        "https://cap3d-um.github.io/",
        author_list("Tiange Luo; Chris Rockwell; Honglak Lee; Justin Johnson"),
        [   Link("Abstract", None, "We introduce Cap3D, an automatic approach for generating descriptive text for 3D objects. This approach utilizes pretrained models from image captioning, image-text alignment, and LLM to consolidate captions from multiple views of a 3D asset, completely side-stepping the time-consuming and costly process of manual annotation. We apply Cap3D to the recently introduced large-scale 3D dataset, Objaverse, resulting in 660k 3D-text pairs. Our evaluation, conducted using 41k human annotations from the same dataset, demonstrates that Cap3D surpasses human-authored descriptions in terms of quality, cost, and speed. Through effective prompt engineering, Cap3D rivals human performance in generating geometric descriptions on 17k collected annotations from the ABO dataset. Finally, we finetune text-to-3D models on Cap3D and human captions, and show Cap3D outperforms; and benchmark the SOTA including Point·E, Shap·E, and DreamFusion." , None),
            Link("Paper", "https://arxiv.org/pdf/2306.07279.pdf", None, None),
            Link("Poster", "posters/Cap3D_poster.pdf", None, None),
        ]
    ),

    Paper(
        "SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation",
        "https://salad3d.github.io/",
        author_list("Juil Koo; Seungwoo Yoo; Hieu Minh Nguyen; Minhyuk Sung"),
        [   Link("Abstract", None, "We present a cascaded diffusion model based on a part-level implicit 3D representation. Our model achieves state-of-the-art generation quality and also enables part-level shape editing and manipulation without any additional training in conditional setup. Diffusion models have demonstrated impressive capabilities in data generation as well as zero-shot completion and editing via a guided reverse process. Recent research on 3D diffusion models has focused on improving their generation capabilities with various data representations, while the absence of structural information has limited their capability in completion and editing tasks. We thus propose our novel diffusion model using a part-level implicit representation. To effectively learn diffusion with high-dimensional embedding vectors of parts, we propose a cascaded framework, learning diffusion first on a low-dimensional subspace encoding extrinsic parameters of parts and then on the other high-dimensional subspace encoding intrinsic attributes. In the experiments, we demonstrate the outperformance of our method compared with the previous ones both in generation and part-level completion and manipulation tasks." , None),
            Link("Paper", "https://arxiv.org/pdf/2303.12236.pdf", None, None),
            Link("Poster", "posters/SALAD-portrait.pdf", None, None),
        ]
    ),

    Paper(
        "Breathing New Life into 3D Assets with Generative Repainting",
        "https://www.obukhov.ai/repainting_3d_assets",
        author_list("Tianfu Wang; Menelaos Kanakis; Konrad Schindler; Luc Van Gool; Anton Obukhov"),
        [   Link("Abstract", None, "Diffusion-based text-to-image models ignited immense attention from the vision community, artists, and content creators. Broad adoption of these models is due to significant improvement in the quality of generations and efficient conditioning on various modalities, not just text. However, lifting the rich generative priors of these 2D models into 3D is challenging. Recent works have proposed various pipelines powered by the entanglement of diffusion models and neural fields. We explore the power of pretrained 2D diffusion models and standard 3D neural radiance fields as independent, standalone tools and demonstrate their ability to work together in a non-learned fashion. Such modularity has the intrinsic advantage of eased partial upgrades, which became an important property in such a fast-paced domain. Our pipeline accepts any legacy renderable geometry, such as textured or untextured meshes, orchestrates the interaction between 2D generative refinement and 3D consistency enforcement tools, and outputs a painted input geometry in several formats. We conduct a large-scale study on a wide range of objects and categories from the ShapeNetSem dataset and demonstrate the advantages of our approach, both qualitatively and quantitatively.", None),
            Link("Paper", "https://www.obukhov.ai/pdf/paper_repainting_3d_assets.pdf", None, None),
        ]
    ),

    Paper(
        "threestudio: a modular framework for diffusion-guided 3D generation",
        "https://github.com/threestudio-project/threestudio",
        author_list("Ying-Tian Liu; Yuan-Chen Guo; Vikram Voleti; Ruizhi Shao; Chia-Hao Chen; Guan Luo; Zixin Zou; Chen Wang; Christian Laforte; Yan-Pei Cao; Song-Hai Zhang"),
        [   Link("Abstract", None, "We introduce threestudio, an open-source, unified, and modular framework specifically designed for 3D content generation. This framework extends diffusion-based 2D image generation models to 3D generation guidance while incorporating conditions such as text and images. We delineate the modular architecture and design of each component within threestudio. Moreover, we re-implement state-of-the-art methods for 3D generation within threestudio, presenting comprehensive comparisons of their design choices. This versatile framework has the potential to empower researchers and developers to delve into cutting-edge techniques for 3D generation, and presents the capability to facilitate further applications beyond 3D generation." , None),
            Link("Paper", "https://cg.cs.tsinghua.edu.cn/threestudio/ICCV2023_AI3DCC_threestudio.pdf", None, None),
            Link("Poster", "posters/threestudio_poster.pdf", None, None),
        ]
    ),

    Paper(
        "Learning Articulated 3D Animals by Distilling 2D Diffusion",
        "https://farm3d.github.io/",
        author_list("Tomas Jakab; Ruining Li; Shangzhe Wu; Christian Rupprecht; Andrea Vedaldi"),
        [   Link("Abstract", None, "We present Farm3D, a method to learn category-specific 3D reconstructors for articulated objects entirely from 'free' virtual supervision from a pre-trained 2D diffusion-based image generator. Recent approaches can learn, given a collection of single-view images of an object category, a monocular network to predict the 3D shape, albedo, illumination and viewpoint of any object occurrence. We propose a framework using an image generator like Stable Diffusion to generate virtual training data for learning such a reconstruction network from scratch. Furthermore, we include the diffusion model as a score to further improve learning. The idea is to randomise some aspects of the reconstruction, such as viewpoint and illumination, generating synthetic views of the reconstructed 3D object, and have the 2D network assess the quality of the resulting image, providing feedback to the reconstructor. Different from work based on distillation which produces a single 3D asset for each textual prompt in hours, our approach produces a monocular reconstruction network that can output a controllable 3D asset from a given image, real or generated, in only seconds. Our network can be used for analysis, including monocular reconstruction, or for synthesis, generating articulated assets for real-time applications such as video games." , None),
            Link("Paper", "https://arxiv.org/pdf/2304.10535.pdf", None, None),
        ]
    ),

    Paper(
        "MeshDiffusion: Score-based Generative 3D Mesh Modeling",
        "https://meshdiffusion.github.io/",
        author_list("Zhen Liu; Yao Feng; Michael J. Black; Derek Nowrouzezahrai; Liam Paull; Weiyang Liu"),
        [   Link("Abstract", None, "We consider the task of generating realistic 3D shapes, which is useful for a variety of applications such as automatic scene generation and physical simulation. Compared to other 3D representations like voxels and point clouds, meshes are more desirable in practice, because (1) they enable easy and arbitrary manipulation of shapes for relighting and simulation, and (2) they can fully leverage the power of modern graphics pipelines which are mostly optimized for meshes. Previous scalable methods for generating meshes typically rely on sub-optimal post-processing, and they tend to produce overly-smooth or noisy surfaces without fine-grained geometric details. To overcome these shortcomings, we take advantage of the graph structure of meshes and use a simple yet very effective generative modeling method to generate 3D meshes. Specifically, we represent meshes with deformable tetrahedral grids, and then train a diffusion model on this direct parametrization. We demonstrate the effectiveness of our model on multiple generative tasks." , None),
            Link("Paper", "https://arxiv.org/pdf/2303.08133.pdf", None, None),
        ]
    ),

    Paper(
        "NCHO: Unsupervised Learning for Neural 3D Composition of Humans and Objects",
        "https://taeksuu.github.io/ncho/",
        author_list("Taeksoo Kim; Shunsuke Saito; Hanbyul Joo"),
        [   Link("Abstract", None, "Deep generative models have been recently extended to synthesizing 3D digital humans. However, previous approaches treat clothed humans as a single chunk of geometry without considering the compositionality of clothing and accessories. As a result, individual items cannot be naturally composed into novel identities, leading to limited expressiveness and controllability of generative 3D avatars. While several methods attempt to address this by leveraging synthetic data, the interaction between humans and objects is not authentic due to the domain gap, and manual asset creation is difficult to scale for a wide variety of objects. In this work, we present a novel framework for learning a compositional generative model of humans and objects (backpacks, coats, scarves, and more) from real-world 3D scans. Our compositional model is interaction-aware, meaning the spatial relationship between humans and objects, and the mutual shape change by physical contact is fully incorporated. The key challenge is that, since humans and objects are in contact, their 3D scans are merged into a single piece. To decompose them without manual annotations, we propose to leverage two sets of 3D scans of a single person with and without objects. Our approach learns to decompose objects and naturally compose them back into a generative human model in an unsupervised manner. Despite our simple setup requiring only the capture of a single subject with objects, our experiments demonstrate the strong generalization of our model by enabling the natural composition of objects to diverse identities in various poses and the composition of multiple objects, which is unseen in training data." , None),
            Link("Paper", "https://arxiv.org/pdf/2305.14345.pdf", None, None),
        ]
    ),

    Paper(
        "Chupa : Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models",
        "https://snuvclab.github.io/chupa/",
        author_list("Byungjun Kim; Patrick Kwon; Kwangho Lee; Myunggi Lee; Sookwan Han; Daesik Kim; Hanbyul Joo"),
        [   Link("Abstract", None, "We propose a 3D generation pipeline that uses diffusion models to generate realistic human digital avatars. Due to the wide variety of human identities, poses, and stochastic details, the generation of 3D human meshes has been a challenging problem. To address this, we decompose the problem into 2D normal map generation and normal map-based 3D reconstruction. Specifically, we first simultaneously generate realistic normal maps for the front and backside of a clothed human, dubbed dual normal maps, using a pose-conditional diffusion model. For 3D reconstruction, we ``carve'' the prior SMPL-X mesh to a detailed 3D mesh according to the normal maps through mesh optimization. To further enhance the high-frequency details, we present a diffusion resampling scheme on both body and facial regions, thus encouraging the generation of realistic digital avatars. We also seamlessly incorporate a recent text-to-image diffusion model to support text-based human identity control. Our method, namely, Chupa, is capable of generating realistic 3D clothed humans with better perceptual quality and identity variety." , None),
            Link("Paper", "https://arxiv.org/pdf/2305.11870.pdf", None, None),
            Link("Poster", "posters/chupa_poster.pdf", None, None),
        ]
    ),

    Paper(
        "CoRF : Colorizing Radiance Fields using Knowledge Distillation",
        "https://arxiv.org/abs/2309.07668",
        author_list("Ankit Dhiman; Srinath R; SrinjaySoumitra Sarkar; Lokesh Boregowda; Venkatesh Babu RADHAKRISHNAN"),
        [   Link("Abstract", None, "Neural radiance field (NeRF) based methods enable high-quality novel-view synthesis for multi-view images. This work presents a method for synthesizing colorized novel views from input grey-scale multi-view images. When we apply image or video-based colorization methods on the generated grey-scale novel views, we observe artifacts due to inconsistency across views. Training a radiance field network on the colorized grey-scale image sequence also does not solve the 3D consistency issue. We propose a distillation based method to transfer color knowledge from the colorization networks trained on natural images to the radiance field network. Specifically, our method uses the radiance field network as a 3D representation and transfers knowledge from existing 2D colorization methods. The experimental results demonstrate that the proposed method produces superior colorized novel views for indoor and outdoor scenes while maintaining cross-view consistency than baselines. Further, we show the efficacy of our method on applications like colorization of radiance field network trained from 1.) Infra-Red (IR) multi-view images and 2.) Old grey-scale multi-view image sequences." , None),
            Link("Paper", "https://arxiv.org/pdf/2309.07668.pdf", None, None),
        ]
    ),

    Paper(
        "CC3D: Layout-Conditioned Generation of Compositional 3D Scenes",
        "https://sherwinbahmani.github.io/cc3d/",
        author_list("Sherwin Bahmani; Jeong Joon Park; Despoina Paschalidou; Xingguang Yan; Gordon Wetzstein; Leonidas Guibas; Andrea Tagliasacchi"),
        [   Link("Abstract", None, "Recent years have seen significant progress in training 3D-aware image generators from unstructured image collections. In this work, we introduce CC3D, a conditional generative model that synthesizes complex 3D scenes conditioned on 2D semantic scene layouts. Different from most existing 3D GANs that limit their applicability to aligned single objects, we focus on generating complex 3D scenes with multiple objects, by modeling the compositional nature of 3D scenes. By devising a 2D layout-based approach for 3D synthesis and implementing a new 3D field representation with a stronger geometric inductive bias, we have created a 3D-GAN that is both efficient and of high quality, while allowing for a more controllable generation process. Our evaluations on synthetic 3D-FRONT and real-world KITTI-360 datasets demonstrate that our model generates scenes of improved visual and geometric quality in comparison to previous works." , None),
            Link("Paper", "https://arxiv.org/pdf/2303.12074.pdf", None, None),
            Link("Poster", "posters/cc3d_poster_iccv3.pdf", None, None),
        ]
    ),

    Paper(
        "Sketch-A-Shape: Zero-Shot Sketch-to-3D Shape Generation",
        "https://arxiv.org/abs/2307.03869",
        author_list("Aditya Sanghi ; Pradeep Kumar Jayaraman; Arianna Rampini; Joseph G Lambourne; Hooman Shayani; Evan Atherton; Saeid Asgari Taghanaki"),
        [   Link("Abstract", None, "Significant progress has been made in using large pre-trained models for creative applications in 3D vision, such as text-to-shape generation. However, generating 3D shapes from sketches remains challenging due to the limited availability of paired datasets linking sketches to corresponding 3D shapes, as well as variations in sketch abstraction levels. To address this challenge, we propose a solution: conditioning a 3D generative model on features extracted from a frozen pre-trained vision model, specifically using features obtained from synthetic renderings during training. This approach enables the effective generation of 3D shapes from sketches at inference time, demonstrating that the pre-trained model features carry semantic signals that are resilient to domain shifts.In our experiments, we validate the effectiveness of our method by showing that it can generate multiple 3D shapes per input sketch, regardless of the sketch's level of abstraction. Importantly, our technique achieves this without the need for paired datasets during training.", None),
            Link("Paper", "https://arxiv.org/pdf/2307.03869.pdf", None, None),
        ]
    ),

    Paper(
        "Locomotion-Action-Manipulation: Synthesizing Human-Scene Interactions in Complex 3D Environments",
        "https://jiyewise.github.io/projects/LAMA/",
        author_list("Jiye Lee; Hanbyul Joo"),
        [   Link("Abstract", None, "Synthesizing interaction-involved human motions has been challenging due to the high complexity of 3D environments and the diversity of possible human behaviors within. We present LAMA, Locomotion-Action-MAnipulation, to synthesize natural and plausible long term human movements in complex indoor environments. The key motivation of LAMA is to build a unified framework to encompass a series of everyday motions including locomotion, scene interaction, and object manipulation. Unlike existing methods that require motion data ''paired'' with scanned 3D scenes for supervision, we formulate the problem as a test-time optimization by using human motion capture data only for synthesis. LAMA leverages a reinforcement learning framework coupled with motion matching algorithm for optimization, and further exploits a motion editing framework via manifold learning to cover possible variations in interaction and manipulation. Throughout extensive experiments, we demonstrate that LAMA outperforms previous approaches in synthesizing realistic motions in various challenging scenarios.", None),
            Link("Paper", "https://arxiv.org/pdf/2301.02667.pdf", None, None),
        ]
    ),

    Paper(
        "Diffusion-based Generation, Optimization, and Planning in 3D Scenes",
        "https://scenediffuser.github.io/",
        author_list("Siyuan Huang; Zan Wang; Puhao Li; Baoxiong Jia; Tengyu Liu; Yixin Zhu; Wei Liang; Song-Chun Zhu"),
        [   Link("Abstract", None, "We introduce the SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior work, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented. With an iterative sampling strategy, SceneDiffuser jointly formulates the scene-aware generation, physics-based optimization, and goal-oriented planning via a diffusion-based denoising process in a fully differentiable fashion. Such a design alleviates the discrepancies among different modules and the posterior collapse of previous scene-conditioned generative models. We evaluate the SceneDiffuser on various 3D scene understanding tasks, including human pose and motion generation, dexterous grasp generation, path planning for 3D navigation, and motion planning for robot arms. The results show significant improvements compared with previous models, demonstrating the tremendous potential of the SceneDiffuser for the broad community of 3D scene understanding.", None),
            Link("Paper", "https://arxiv.org/pdf/2301.06015.pdf", None, None),
        ]
    ),

    Paper(
        "Text-driven Human Avatar Generation by Neural Re-parameterized Texture Optimization",
        "",
        author_list("Kim Youwang; Tae-Hyun Oh"),
        [   Link("Abstract", None, "We present TexAvatar, a text-driven human texture generation system for creative human avatar synthesis. Despite the huge progress in text-driven human avatar generation methods, modeling high-quality, efficient human appearance remains challenging. With our proposed neural re-parameterized texture optimization, TexAvatar generates a high-quality UV texture in 30 minutes, given only a text description. The generated UV texture can be easily superimposed on animatable human meshes without further processing. This is distinctive in that prior works generate volumetric textured avatars that require cumbersome rigging processes to animate. We demonstrate that TexAvatar produces human avatars with favorable quality, with faster speed, compared to recent competing methods.", None),
            Link("Paper", "/papers/0042_camready.pdf", None, None),
            Link("Poster", "posters/0042_poster.pdf", None, None),
        ]
    ),

    Paper(
        "InterDiff: Forecasting 3D Human-Object Interaction with Physics-Informed Diffusion",
        "https://sirui-xu.github.io/InterDiff/",
        author_list("Sirui Xu; Zhengyuan Li; Yu-Xiong Wang; Liangyan Gui"),
        [   Link("Abstract", None, "This paper addresses the task of anticipating 3D human-object interactions (HOIs). Previous research has either ignored object dynamics or lacked whole-body interaction being limited to grasping small objects. Our task is challenging, as it requires modeling dynamic objects with various shapes, whole-body motion, and ensuring physically valid interaction. To this end, we propose InterDiff, a framework comprising two key steps: (i) interaction diffusion, where we leverage a diffusion model to capture the distribution of future human-object interactions; (ii) interaction correction, where we introduce a physics-informed predictor via coordinate transformations to correct for denoised HOIs in a diffusion step. Our key insight is to inject prior knowledge that the interactions under reference with respect to the contact points follow a simple pattern and are easily predictable. Experiments on large-scale human motion datasets demonstrate the effectiveness of our method for the new task, capable of producing realistic, vivid, and remarkably long-term 3D HOI predictions.", None),
            Link("Paper", "https://arxiv.org/pdf/2308.16905.pdf", None, None),
        ]
    ),

    Paper(
        "Locally Stylized Neural Radiance Fields",
        "https://arxiv.org/abs/2309.10684",
        author_list("Hong Wing Pang; Binh-Son Hua; Sai-Kit Yeung"),
        [   Link("Abstract", None, "In recent years, there has been increasing interest in applying stylization on 3D scenes from a reference style image, in particular onto neural radiance fields (NeRF). While performing stylization directly on NeRF guarantees appearance consistency over arbitrary novel views, it is a challenging problem to guide the transfer of patterns from the style image onto different parts of the NeRF scene. In this work, we propose a stylization framework for NeRF based on local style transfer. In particular, we use a hash-grid encoding to learn the embedding of the appearance and geometry components, and show that the mapping defined by the hash table allows us to control the stylization to a certain extent. Stylization is then achieved by optimizing the appearance branch while keeping the geometry branch fixed. To support local style transfer, we propose a new loss function that utilizes a segmentation network and bipartite matching to establish region correspondences between the style image and the content images obtained from volume rendering. Our experiments show that our method yields plausible stylization results with novel view synthesis while having flexible controllability via manipulating and customizing the region correspondences.", None),
            Link("Paper", "https://arxiv.org/pdf/2309.10684.pdf", None, None),
            Link("Poster", "posters/nerf_stylization_poster.pdf", None, None),
        ]
    ),

    Paper(
        "ContactGen: Generative Contact Modeling for Grasp Generation",
        "#",
        author_list("Shaowei Liu; Yang Zhou; Jimei Yang"),
        [   Link("Abstract", None, "This paper presents a novel object-centric contact representation ContactGen for hand-object interaction. The ContactGen comprises 3 components: a contact map indicates the contact location, a part map represents the contact hand part, and a direction map tells the contact direction within each part. Given an input object, we propose a conditional generative model to predict ContactGen and adopt model-based optimization to predict diverse and geometrically feasible grasps. Experimental results demonstrate our method can generate high-fidelity and diverse human grasps for various objects.", None),
            Link("Paper", "#", None, None),
        ]
    ),
]


def build_publications_list(publications):
    def image(paper):
        if paper.image is not None:
            return '<img src="{}" alt="{}" />'.format(
                paper.image, paper.title
            )
        else:
            return '&nbsp;'

    def title(paper):
        return '<a href="{}">{}</a>'.format(paper.url, paper.title)

    def authors(paper):
        return ", ".join(a for a in paper.authors)

    def links(paper):
        def links_list(paper):
            def link(i, link):
                if link.url is not None:
                    # return '<a href="{}">{}</a>'.format(link.url, link.name)
                    return '<a href="{}" data-type="{}">{}</a>'.format(link.url, link.name, link.name)
                else:
                    return '<a href="#" data-type="{}" data-index="{}">{}</a>'.format(link.name, i, link.name)
            return " ".join(
                link(i, l) for i, l in enumerate(paper.links)
            )

        def links_content(paper):
            def content(i, link):
                if link.url is not None:
                    return ""
                return '<div class="link-content" data-index="{}">{}</div>'.format(
                    i, link.html if link.html is not None
                       else '<pre>' + link.text + "</pre>"
                )
            return "".join(content(i, link) for i, link in enumerate(paper.links))
        return links_list(paper) + links_content(paper)

    def paper(p):
        return ('<div class="row paper">'
                    '<div class="content">'
                        '<div class="paper-title">{}</div>'
                        '<div class="authors">{}</div>'
                        '<div class="links">{}</div>'
                    '</div>'
                '</div>').format(
                title(p),
                authors(p),
                links(p)
            )

    return "".join(paper(p) for p in publications)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Create a publication list and insert in into an html file"
    )
    parser.add_argument(
        "file",
        help="The html file to insert the publications to"
    )

    parser.add_argument(
        "--safe", "-s",
        action="store_true",
        help="Do not overwrite the file but create one with suffix .new"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Do not output anything to stdout/stderr"
    )

    args = parser.parse_args(argv)

    # Read the file
    with open(args.file) as f:
        html = f.read()

    # Find the fence comments
    start_text = "<!-- start non-arc paper list -->"
    end_text = "<!-- end non-arc paper list -->"
    start = html.find(start_text)
    end = html.find(end_text, start)
    if end < start or start < 0:
        _print(args, "Could not find the fence comments", file=sys.stderr)
        sys.exit(1)

    # Build the publication list in html
    replacement = build_publications_list(publications)

    # Update the html and save it
    html = html[:start+len(start_text)] + replacement + html[end:]

    # If safe is set do not overwrite the input file
    if args.safe:
        with open(args.file + ".new", "w") as f:
            f.write(html)
    else:
        with open(args.file, "w") as f:
            f.write(html)


if __name__ == "__main__":
    main(None)

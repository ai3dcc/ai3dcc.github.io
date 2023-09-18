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
        ]
    ),

    Paper(
        "BallGAN: 3D-aware Image Synthesis with a Spherical Background",
        "https://minjung-s.github.io/ballgan",
        author_list("Minjung Shin; Yunji Seo; Jeongmin Bae; Young Sun Choi; Hyunsu Kim; Hyeran Byun; Youngjung Uh"),
        [   Link("Abstract", None, "3D-aware GANs aim to synthesize realistic 3D scenes that can be rendered in arbitrary camera viewpoints, generating high-quality images with well-defined geometry. As 3D content creation becomes more popular, the ability to generate foreground objects separately from the background has become a crucial property. Existing methods have been developed regarding overall image quality, but they can not generate foreground objects only and often show degraded 3D geometry. In this work, we propose to represent the background as a spherical surface for multiple reasons inspired by computer graphics. Our method naturally provides foreground-only 3D synthesis facilitating easier 3D content creation. Furthermore, it improves the foreground geometry of 3D-aware GANs and the training stability on datasets with complex backgrounds.", None),
            Link("Paper", "https://arxiv.org/pdf/2301.09091.pdf", None, None),
        ]
    ),

    Paper(
        "3D-aware Blending with Generative NeRFs",
        "https://blandocs.github.io/blendnerf",
        author_list("Hyunsu Kim; Gayoung Lee; Yunjey Choi; Jin-Hwa Kim); Jun-Yan Zhu"),
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
        ]
    ),

    Paper(
        "SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation",
        "https://salad3d.github.io/",
        author_list("Juil Koo; Seungwoo Yoo; Hieu Minh Nguyen; Minhyuk Sung"),
        [   Link("Abstract", None, "We present a cascaded diffusion model based on a part-level implicit 3D representation. Our model achieves state-of-the-art generation quality and also enables part-level shape editing and manipulation without any additional training in conditional setup. Diffusion models have demonstrated impressive capabilities in data generation as well as zero-shot completion and editing via a guided reverse process. Recent research on 3D diffusion models has focused on improving their generation capabilities with various data representations, while the absence of structural information has limited their capability in completion and editing tasks. We thus propose our novel diffusion model using a part-level implicit representation. To effectively learn diffusion with high-dimensional embedding vectors of parts, we propose a cascaded framework, learning diffusion first on a low-dimensional subspace encoding extrinsic parameters of parts and then on the other high-dimensional subspace encoding intrinsic attributes. In the experiments, we demonstrate the outperformance of our method compared with the previous ones both in generation and part-level completion and manipulation tasks." , None),
            Link("Paper", "https://arxiv.org/pdf/2303.12236.pdf", None, None),
        ]
    ),

    Paper(
        "Breathing New Life into 3D Assets with Generative Repainting",
        "https://www.obukhov.ai/repainting_3d_assets",
        author_list("Tianfu Wang; Menelaos Kanakis; Konrad Schindler; Luc Van Gool; Anton Obukhov"),
        [   Link("Abstract", None, "We present a cascaded diffusion model based on a part-level implicit 3D representation. Our model achieves state-of-the-art generation quality and also enables part-level shape editing and manipulation without any additional training in conditional setup. Diffusion models have demonstrated impressive capabilities in data generation as well as zero-shot completion and editing via a guided reverse process. Recent research on 3D diffusion models has focused on improving their generation capabilities with various data representations, while the absence of structural information has limited their capability in completion and editing tasks. We thus propose our novel diffusion model using a part-level implicit representation. To effectively learn diffusion with high-dimensional embedding vectors of parts, we propose a cascaded framework, learning diffusion first on a low-dimensional subspace encoding extrinsic parameters of parts and then on the other high-dimensional subspace encoding intrinsic attributes. In the experiments, we demonstrate the outperformance of our method compared with the previous ones both in generation and part-level completion and manipulation tasks." , None),
            Link("Paper", "https://www.obukhov.ai/pdf/paper_repainting_3d_assets.pdf", None, None),
        ]
    ),

    Paper(
        "threestudio: a modular framework for diffusion-guided 3D generation",
        "https://github.com/threestudio-project/threestudio",
        author_list("Ying-Tian Liu; Yuan-Chen Guo; Vikram Voleti; Ruizhi Shao; Chia-Hao Chen; Guan Luo; Zixin Zou; Chen Wang; Christian Laforte; Yan-Pei Cao; Song-Hai Zhang"),
        [   Link("Abstract", None, "We introduce threestudio, an open-source, unified, and modular framework specifically designed for 3D content generation. This framework extends diffusion-based 2D image generation models to 3D generation guidance while incorporating conditions such as text and images. We delineate the modular architecture and design of each component within threestudio. Moreover, we re-implement state-of-the-art methods for 3D generation within threestudio, presenting comprehensive comparisons of their design choices. This versatile framework has the potential to empower researchers and developers to delve into cutting-edge techniques for 3D generation, and presents the capability to facilitate further applications beyond 3D generation." , None),
            Link("Paper", "https://cg.cs.tsinghua.edu.cn/threestudio/ICCV2023_AI3DCC_threestudio.pdf", None, None),
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

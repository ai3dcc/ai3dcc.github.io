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
        "Looking at Words and Points with Attention: a Benchmark for Text-to-Shape Coherence",
        "https://arxiv.org/abs/2309.07917",
        author_list("Andrea Amaduzzi; Giuseppe Lisanti; Samuele Salti; Luigi Di Stefano"),
        [   Link("Abstract", None, "While text-conditional 3D object generation and manipulation have seen rapid progress, the evaluation of coherence between generated 3D shapes and input textual descriptions lacks a clear benchmark. The reason is twofold: a) the low quality of the textual descriptions in the only publicly available dataset of text-shapes pairs; b) the limited effectiveness of the metrics used to quantitatively assess such coherence. In this paper, we propose a comprehensive solution that addresses both weaknesses. Firstly, we employ large language models to automatically refine textual descriptions associated with shapes. Secondly, we propose a quantitative metric to assess text-to-shape coherence, through cross-attention mechanism. To validate our approach, we conduct a human study and compare quantitatively our metric with existing ones. The refined dataset, the new metric and a set of text-shape pairs validated by the human study comprise a novel, fine-grained benchmark that we publicly release to foster research on text-to-shape coherence of text-conditioned 3D generative models.", None),
            Link("Paper", "https://arxiv.org/pdf/2309.07917.pdf", None, None),
            Link("Poster", "posters/full_poster_Andrea_Amaduzzi.pdf", None, None),
        ]
    ),

    Paper(
        "BluNF: Blueprint Neural Field",
        "https://www.lix.polytechnique.fr/vista/projects/2023_iccvw_courant/",
        author_list("Robin Courant; Xi WANG; Marc Christie; Vicky Kalogeiton"),
        [   Link("Abstract", None, "Neural Radiance Fields (NeRFs) have revolutionized scene novel view synthesis, offering visually realistic, precise, and robust implicit reconstructions. While recent approaches enable NeRF editing, such as object removal, 3D shape modification, or material property manipulation, the manual annotation prior to such edits makes the process tedious. Additionally, traditional 2D interaction tools lack an accurate sense of 3D space, preventing precise manipulation and editing of scenes. In this paper, we introduce a novel approach, called Blueprint Neural Field (BluNF), to address these editing issues. BluNF provides a robust and user-friendly 2D blueprint, enabling intuitive scene editing. By leveraging implicit neural representation, BluNF constructs a blueprint of a scene using prior semantic and depth information. The generated blueprint allows effortless editing and manipulation of NeRF representations. We demonstrate BluNF's editability through an intuitive click-and-change mechanism, enabling 3D manipulations, such as masking, appearance modification, and object removal. Our approach significantly contributes to visual content creation, paving the way for further research in this area.", None),
            Link("Paper", "https://arxiv.org/pdf/2309.03933.pdf", None, None),
            Link("Poster", "posters/BluNF-poster.pdf", None, None),
        ]
    ),

    Paper(
        "NeRF-GAN Distillation for Efficient 3D-Aware Generation with Convolutions",
        "https://arxiv.org/abs/2303.12865",
        author_list("Shahbazi Mohamad; Ntavelis Evangelos; Tonioni Alessio; Collins Edo; Paudel Danda Pani; Danelljan Martin; Van Gool Luc"),
        [   Link("Abstract", None, "Pose-conditioned convolutional generative models struggle with high-quality 3D-consistent image generation from single-view datasets, due to their lack of sufficient 3D priors. Recently, the integration of Neural Radiance Fields (NeRFs) and generative models, such as Generative Adversarial Networks (GANs), has transformed 3D-aware generation from single-view images. NeRF-GANs exploit the strong inductive bias of neural 3D representations and volumetric rendering at the cost of higher computational complexity. This study aims at revisiting pose-conditioned 2D GANs for efficient 3D-aware generation at inference time by distilling 3D knowledge from pretrained NeRF-GANs. We propose a simple and effective method, based on re-using the well-disentangled latent space of a pre-trained NeRF-GAN in a pose-conditioned convolutional network to directly generate 3D-consistent images corresponding to the underlying 3D representations. Experiments on several datasets demonstrate that the proposed method obtains results comparable with volumetric rendering in terms of quality and 3D consistency while benefiting from the computational advantage of convolutional networks.", None),
            Link("Paper", "https://arxiv.org/pdf/2303.12865.pdf", None, None),
            Link("Poster", "posters/ICCVW2023_NeRF-GAN_Poster.pdf", None, None),
        ]
    ),

    Paper(
        "LatentSwap3D: Semantic Edits on 3D Image GANs",
        "https://arxiv.org/abs/2212.01381",
        author_list("Enis Simsar; Alessio Tonioni; Evin Pınar Örnek; Federico Tombari"),
        [   Link("Abstract", None, "Recent 3D GANs have the ability to generate latent codes for entire 3D volumes rather than only 2D images. While they offer desirable features like high-quality geometry and multi-view consistency, complex semantic image editing tasks for 3D GANs have only been partially explored, unlike their 2D counterparts, e.g., StyleGAN and its variants. To address this problem, we propose LatentSwap3D, a latent space discovery-based semantic edit approach which can be used with any off-the-shelf 3D or 2D GAN model and on any dataset. LatentSwap3D relies on identifying the latent code dimensions corresponding to specific attributes by feature ranking of a random forest classifier. It then performs the edit by swapping the selected dimensions of the image being edited with the ones from an automatically selected reference image. Compared to other latent space control-based edit methods, which were mainly designed for 2D GANs, our method on 3D GANs provides remarkably consistent semantic edits in a disentangled manner and outperforms others both qualitatively and quantitatively. We show results on seven 3D generative models (pi-GAN, GIRAFFE, StyleSDF, MVCGAN, EG3D, StyleNeRF, and VolumeGAN) and on five datasets (FFHQ, AFHQ, Cats, MetFaces, and CompCars).", None),
            Link("Paper", "https://arxiv.org/pdf/2212.01381.pdf", None, None),
            Link("Poster", "posters/AI3DCC-LatentSwap3D.pdf", None, None),
        ]
    ),

    Paper(
        "BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models",
        "https://arxiv.org/abs/2309.00158",
        author_list("Yao Wei; George Vosselman; Michael Ying Yang"),
        [   Link("Abstract", None, "3D building generation with low data acquisition costs, such as single image-to-3D, becomes increasingly important. However, most of the existing single image-to-3D building creation works are restricted to those images with specific viewing angles, hence they are difficult to scale to general-view images that commonly appear in practical cases. To fill this gap, we propose a novel 3D building shape generation method exploiting point cloud diffusion models with image conditioning schemes, which demonstrates flexibility to the input images. By cooperating two conditional diffusion models and introducing a regularization strategy during denoising process, our method is able to synthesize building roofs while maintaining the overall structures. We validate our framework on two newly built datasets and extensive experiments show that our method outperforms previous works in terms of building generation quality.", None),
            Link("Paper", "https://arxiv.org/pdf/2309.00158.pdf", None, None),
            Link("Poster", "posters/ICCV23_AI3DCC_BuilDiff.pdf", None, None),
        ]
    ),

    Paper(
        "Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes",
        "https://arxiv.org/abs/2303.13450",
        author_list("Dana Cohen-Bar; Elad Richardson; Gal Metzer; Raja Giryes; Danny Cohen-Or"),
        [   Link("Abstract", None, "Recent breakthroughs in text-guided image generation have led to remarkable progress in the field of 3D synthesis from text. By optimizing neural radiance fields (NeRF) directly from text, recent methods are able to produce remarkable results. Yet, these methods are limited in their control of each object's placement or appearance, as they represent the scene as a whole. This can be a major issue in scenarios that require refining or manipulating objects in the scene. To remedy this deficit, we propose a novel Global-Local training framework for synthesizing a 3D scene using object proxies. A proxy represents the object's placement in the generated scene and optionally defines its coarse geometry. The key to our approach is to represent each object as an independent NeRF. We alternate between optimizing each NeRF on its own and as part of the full scene. Thus, a complete representation of each object can be learned, while also creating a harmonious scene with style and lighting match. We show that using proxies allows a wide variety of editing options, such as adjusting the placement of each independent object, removing objects from a scene, or refining an object. Our results show that Set-the-Scene offers a powerful solution for scene synthesis and manipulation, filling a crucial gap in controllable text-to-3D synthesis.", None),
            Link("Paper", "https://arxiv.org/pdf/2303.13450.pdf", None, None),
            Link("Poster", "posters/set_the_scene.pdf", None, None),
        ]
    ),

    Paper(
        "SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images",
        "https://abdullahamdi.com/sparf/",
        author_list("Abdullah J Hamdi; Bernard Ghanem; Matthias Niessner"),
        [   Link("Abstract", None, "Recent advances in Neural Radiance Fields (NeRFs) treat the problem of novel view synthesis as Sparse Radiance Field (SRF) optimization using sparse voxels for efficient and fast rendering (plenoxels,INGP). In order to leverage machine learning and adoption of SRFs as a 3D representation, we present SPARF, a large-scale ShapeNet-based synthetic dataset for novel view synthesis consisting of ~ 17 million images rendered from nearly 40,000 shapes at high resolution (400 * 400 pixels). The dataset is orders of magnitude larger than existing synthetic datasets for novel view synthesis and includes more than one million 3D-optimized radiance fields with multiple voxel resolutions. Furthermore, we propose a novel pipeline (SuRFNet) that learns to generate sparse voxel radiance fields from only few views. This is done by using the densely collected SPARF dataset and 3D sparse convolutions. SuRFNet employs partial SRFs from few/one images and a specialized SRF loss to learn to generate high-quality sparse voxel radiance fields that can be rendered from novel views. Our approach achieves state-of-the-art results in the task of unconstrained novel view synthesis based on few views on ShapeNet as compared to recent baselines. The SPARF dataset will be made publicly available with the code and models upon publication.", None),
            Link("Paper", "https://arxiv.org/pdf/2212.09100.pdf", None, None),
            Link("Poster", "posters/SPARF_poster.pdf", None, None),
        ]
    ),

    Paper(
        "Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields",
        "https://www.vision.huji.ac.il/blended-nerf/",
        author_list("Ori Gordon; Omri Avrahami; Dani Lischinski"),
        [   Link("Abstract", None, "Editing a local region or a specific object in a 3D scene represented by a NeRF or consistently blending a new realistic object into the scene is challenging, mainly due to the implicit nature of the scene representation. We present Blended-NeRF, a robust and flexible framework for editing a specific region of interest in an existing NeRF scene, based on text prompts, along with a 3D ROI box. Our method leverages a pretrained language-image model to steer the synthesis towards a user-provided text prompt, along with a 3D MLP model initialized on an existing NeRF scene to generate the object and blend it into a specified region in the original scene. We allow local editing by localizing a 3D ROI box in the input scene, and blend the content synthesized inside the ROI with the existing scene using a novel volumetric blending technique. To obtain natural looking and view-consistent results, we leverage existing and new geometric priors and 3D augmentations for improving the visual fidelity of the final result. We test our framework both qualitatively and quantitatively on a variety of real 3D scenes and text prompts, demonstrating realistic multi-view consistent results with much flexibility and diversity compared to the baselines. Finally, we show the applicability of our framework for several 3D editing applications, including adding new objects to a scene, removing/replacing/altering existing objects, and texture conversion.", None),
            Link("Paper", "https://arxiv.org/pdf/2306.12760.pdf", None, None),
            Link("Poster", "posters/BlendedNeRF.pdf", None, None),
        ]
    ),

    Paper(
        "S2RF: Semantically Stylized Radiance Fields",
        "https://arxiv.org/abs/2309.01252",
        author_list("Moneish Kumar; Neeraj Panse; Dishani Lahiri"),
        [   Link("Abstract", None, "We present our method for transferring style from any arbitrary image(s) to object(s) within a 3D scene. Our primary objective is to offer more control in 3D scene stylization, facilitating the creation of customizable and stylized scene images from arbitrary viewpoints. To achieve this, we propose a novel approach that incorporates nearest neighborhood-based loss, allowing for flexible 3D scene reconstruction while effectively capturing intricate style details and ensuring multi-view consistency.", None),
            Link("Paper", "https://arxiv.org/pdf/2309.01252.pdf", None, None),
            Link("Poster", "posters/ICCV-Poster-S2RF.pdf", None, None),
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
    start_text = "<!-- start paper list -->"
    end_text = "<!-- end paper list -->"
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

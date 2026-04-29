// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Visual Jenga',
  tagline: 'Discovering Object Dependencies via Counterfactual Inpainting',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://everloom-129.github.io',
  baseUrl: '/visualjenga/',

  organizationName: 'Everloom-129',
  projectName: 'visualjenga',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl:
            'https://github.com/Everloom-129/visualjenga/tree/main/docs-site/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      navbar: {
        title: 'Visual Jenga',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'mainSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'https://arxiv.org/abs/2503.21770',
            label: 'Paper',
            position: 'right',
          },
          {
            href: 'https://github.com/Everloom-129/visualjenga',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Reproduction',
                to: '/docs/reproduction',
              },
            ],
          },
          {
            title: 'Links',
            items: [
              {
                label: 'Original Paper (arXiv)',
                href: 'https://arxiv.org/abs/2503.21770',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/Everloom-129/visualjenga',
              },
              {
                label: 'Jie Wang',
                href: 'https://everloom-129.github.io/',
              },
            ],
          },
        ],
        copyright: `Copyright \u00a9 ${new Date().getFullYear()} Jie Wang. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'python'],
      },
    }),
};

export default config;

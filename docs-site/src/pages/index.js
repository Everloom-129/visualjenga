import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/reproduction">
            View Reproduction Results
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Visual Jenga — discovering object dependencies via counterfactual inpainting. Reproduction by Jie Wang.">
      <HomepageHeader />
      <main>
        <section style={{padding: '2rem 0', textAlign: 'center'}}>
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <Heading as="h2">About This Project</Heading>
                <p style={{fontSize: '1.1rem'}}>
                  Visual Jenga progressively removes objects from a scene to reveal
                  implicit physical dependencies. Given a single image, the pipeline
                  iteratively identifies and removes the "most removable" object
                  using counterfactual inpainting until only the background remains.
                </p>
                <p>
                  <a href="https://arxiv.org/abs/2503.21770">Original Paper (arXiv)</a>
                  {' | '}
                  <a href="https://github.com/Everloom-129/visualjenga">GitHub Repo</a>
                  {' | '}
                  <a href="https://everloom-129.github.io/">Jie Wang</a>
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

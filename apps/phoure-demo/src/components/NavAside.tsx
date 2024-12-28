import { CircleHelpIcon } from './ui/circle-help';
import { HoverCard, HoverCardContent, HoverCardTrigger } from './ui/hover-card';
import { GithubIcon } from './ui/github';

export function NavAside() {
  return (
    <aside className="absolute top-2 right-2 text-black">
      <nav>
        <ul className="flex flex-row gap-2">
          <li>
            <HoverCard openDelay={200} closeDelay={400}>
              <HoverCardTrigger>
                <CircleHelpIcon />
              </HoverCardTrigger>
              <HoverCardContent>
                <p className="text-sm text-justify">
                  The goal of <strong>phoure</strong> is to be an upscaling
                  solution that does not infringe on the rights of artists. Try
                  it out in this live demo.
                </p>
              </HoverCardContent>
            </HoverCard>
          </li>
          <li>
            <a
              href="https://github.com/iwoplaza/phoure"
              target="_blank"
              rel="noreferrer noopener"
            >
              <GithubIcon />
            </a>
          </li>
        </ul>
      </nav>
    </aside>
  );
}

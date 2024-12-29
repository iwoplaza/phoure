import type { CheckedState } from '@radix-ui/react-checkbox';
import type { SliderProps } from '@radix-ui/react-slider';
import {
  type SetStateAction,
  type WritableAtom,
  useAtom,
  useSetAtom,
} from 'jotai';
import type { RESET } from 'jotai/utils';
import { ChevronDown } from 'lucide-react';
import { useCallback, useId } from 'react';

import { accumulatedLayersAtom } from '@/GameEngine/sdfRenderer/sdfRenderer';
import {
  type DisplayMode,
  DisplayModes,
  autoRotateControlAtom,
  autoRotateSpeedAtom,
  cameraFovControlAtom,
  cameraOrientationControlAtom,
  cameraYControlAtom,
  cameraZoomControlAtom,
  displayModeAtom,
  fixedTimestepAtom,
  fixedTimestepEnabledAtom,
  targetResolutionAtom,
} from '@/controlAtoms';
import { Checkbox } from './ui/checkbox';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import { Label } from './ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from './ui/sidebar';
import { Slider } from './ui/slider';

function ControlLabel(props: { htmlFor: string; children: string }) {
  return (
    <div className="min-h-[2rem] flex items-center">
      <Label className="text-xs text-slate-500" htmlFor={props.htmlFor}>
        {props.children}
      </Label>
    </div>
  );
}

function SliderControl(
  props: {
    label: string;
    valueAtom: WritableAtom<
      number,
      [SetStateAction<number | typeof RESET>],
      void
    >;
  } & SliderProps,
) {
  const { label, valueAtom, ...rest } = props;

  const id = useId();
  const [value, setValue] = useAtom(valueAtom);
  const setAccumulatedLayers = useSetAtom(accumulatedLayersAtom);

  const onValueChange = useCallback(
    (values: number[]) => {
      setValue(values[0]);
      setAccumulatedLayers(0);
    },
    [setValue, setAccumulatedLayers],
  );

  return (
    <>
      <div className="px-4 my-2">
        <ControlLabel htmlFor={id}>{label}</ControlLabel>
        <div className="flex justify-self-stretch gap-2">
          <Slider
            {...rest}
            value={[value]}
            onValueChange={onValueChange}
            className="w-[180px]"
            id={id}
          />
          <p className="min-w-12 text-right">{value}</p>
        </div>
      </div>
    </>
  );
}

function CheckboxControl(props: {
  label: string;
  valueAtom: WritableAtom<
    boolean,
    // biome-ignore lint/suspicious/noExplicitAny: <does not really matter>
    [boolean | any],
    void
  >;
}) {
  const { label, valueAtom } = props;

  const id = useId();
  const [checked, setChecked] = useAtom(valueAtom);

  const onCheckedChange = useCallback(
    (e: CheckedState) => {
      setChecked(e === true);
    },
    [setChecked],
  );

  return (
    <div className="px-4 my-2 flex items-center justify-between">
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
      <Checkbox checked={checked} onCheckedChange={onCheckedChange} id="id" />
    </div>
  );
}

function DisplayModeControl() {
  const [displayMode, setDisplayMode] = useAtom(displayModeAtom);
  const setAccumulatedLayers = useSetAtom(accumulatedLayersAtom);

  const onValueChange = useCallback(
    (value: string) => {
      setDisplayMode(value as DisplayMode);
      setAccumulatedLayers(0);
    },
    [setDisplayMode, setAccumulatedLayers],
  );

  return (
    <div className="px-4 my-2">
      <ControlLabel htmlFor="display-mode">Display mode</ControlLabel>
      <Select value={displayMode} onValueChange={onValueChange}>
        <SelectTrigger className="w-[180px]" id="display-mode">
          <SelectValue placeholder="Display mode" />
        </SelectTrigger>
        <SelectContent>
          {DisplayModes.map((mode) => (
            <SelectItem key={mode.key} value={mode.key}>
              {mode.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function TargetResolutionControl() {
  const [targetResolution, setTargetResolution] = useAtom(targetResolutionAtom);
  const setAccumulatedLayers = useSetAtom(accumulatedLayersAtom);

  const onValueChange = useCallback(
    (value: string) => {
      setTargetResolution(Number.parseInt(value));
      setAccumulatedLayers(0);
    },
    [setTargetResolution, setAccumulatedLayers],
  );

  return (
    <div className="px-4">
      <ControlLabel htmlFor="target-resolution">Target resolution</ControlLabel>
      <Select value={String(targetResolution)} onValueChange={onValueChange}>
        <SelectTrigger className="w-[180px]" id="target-resolution">
          <SelectValue placeholder="Target resolution" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={'256'}>256x256</SelectItem>
          <SelectItem value={'360'}>360x360</SelectItem>
          <SelectItem value={'480'}>480x480</SelectItem>
          <SelectItem value={'512'}>512x512</SelectItem>
          <SelectItem value={'750'}>750x750</SelectItem>
          <SelectItem value={'1024'}>1024x1024</SelectItem>
          <SelectItem value={'1600'}>1600x1600</SelectItem>
          <SelectItem value={'2048'}>2048x2048</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

interface ControlGroupProps {
  label: string;
  children: React.ReactNode;
}

function ControlGroup(props: ControlGroupProps) {
  const { label, children } = props;

  return (
    <Collapsible defaultOpen className="group/collapsible">
      <SidebarGroup>
        <SidebarGroupLabel asChild>
          <CollapsibleTrigger>
            {label}
            <ChevronDown className="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-180" />
          </CollapsibleTrigger>
        </SidebarGroupLabel>
        <CollapsibleContent>
          <SidebarGroupContent>{children}</SidebarGroupContent>
        </CollapsibleContent>
      </SidebarGroup>
    </Collapsible>
  );
}

export function ControlsSidebar() {
  return (
    <Sidebar side="left" variant="sidebar">
      <SidebarContent>
        <ControlGroup label="Rendering">
          <DisplayModeControl />
          <TargetResolutionControl />
        </ControlGroup>
        <ControlGroup label="Time controls">
          <CheckboxControl
            label="Fixed timestep"
            valueAtom={fixedTimestepEnabledAtom}
          />
          <SliderControl
            label="Seconds per frame"
            valueAtom={fixedTimestepAtom}
            min={0.1}
            step={0.1}
            max={2}
          />
        </ControlGroup>
        <ControlGroup label="Camera controls">
          <SliderControl
            label="Camera orientation"
            valueAtom={cameraOrientationControlAtom}
            max={360}
          />
          <SliderControl
            label="Camera Y"
            valueAtom={cameraYControlAtom}
            min={-0.2}
            step={0.01}
            max={1}
          />
          <SliderControl
            label="Camera Zoom"
            valueAtom={cameraZoomControlAtom}
            min={1}
            step={0.01}
            max={4}
          />
          <SliderControl
            label="Camera FOV"
            valueAtom={cameraFovControlAtom}
            min={20}
            step={1}
            max={170}
          />
          <CheckboxControl
            label="Auto rotate"
            valueAtom={autoRotateControlAtom}
          />
          <SliderControl
            label="Auto rotate speed"
            valueAtom={autoRotateSpeedAtom}
            min={0.5}
            step={0.01}
            max={100}
          />
        </ControlGroup>
      </SidebarContent>
    </Sidebar>
  );
}

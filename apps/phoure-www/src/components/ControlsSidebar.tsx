import type { CheckedState } from '@radix-ui/react-checkbox';
import type { SliderProps } from '@radix-ui/react-slider';
import { useAtom, useSetAtom, type WritableAtom } from 'jotai';
import { ChevronDown } from 'lucide-react';
import { useCallback, useId } from 'react';

import { Checkbox } from 'src/components/ui/checkbox';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from 'src/components/ui/collapsible';
import { Label } from 'src/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from 'src/components/ui/select';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from 'src/components/ui/sidebar';
import { Slider } from 'src/components/ui/slider';
import { accumulatedLayersAtom } from 'src/lib/GameEngine/sdfRenderer/sdfRenderer.ts';
import {
  autoRotateControlAtom,
  autoRotateSpeedAtom,
  cameraFovControlAtom,
  cameraOrientationControlAtom,
  cameraYControlAtom,
  cameraZoomControlAtom,
  type DisplayMode,
  displayModeAtom,
  DisplayModes,
  fixedTimestepAtom,
  fixedTimestepEnabledAtom,
  targetResolutionAtom,
} from 'src/lib/controlAtoms.ts';
import { Separator } from './ui/separator';

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
    valueAtom: WritableAtom<number, [number], void>;
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
      <div className="px-4 mt-2 mb-4">
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
  valueAtom: WritableAtom<boolean, [boolean], void>;
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
    <div className="px-4 mt-4 flex items-center justify-between">
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
        <Separator />
        <ControlGroup label="Time">
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
        <Separator />
        <ControlGroup label="Camera">
          <SliderControl
            label="Up/down position"
            valueAtom={cameraYControlAtom}
            min={-0.2}
            step={0.01}
            max={1}
          />
          <SliderControl
            label="Distance from origin"
            valueAtom={cameraZoomControlAtom}
            min={1}
            step={0.01}
            max={4}
          />
          <SliderControl
            label="Field of view"
            valueAtom={cameraFovControlAtom}
            min={20}
            step={1}
            max={170}
          />
          <SliderControl
            label="Yaw angle"
            valueAtom={cameraOrientationControlAtom}
            max={360}
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

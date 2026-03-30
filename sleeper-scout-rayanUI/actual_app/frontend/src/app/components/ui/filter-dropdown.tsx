"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { ChevronUp, Check, Filter, Search } from "lucide-react"
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export interface Option {
  label: string
  value: string | number
  description?: string
}

interface FilterDropdownProps {
  value: string | number
  options: Option[]
  onChange: (value: any) => void
  icon?: React.ReactNode
  label: string
  className?: string
  accentColor?: string
  colorAllOptions?: boolean
  searchable?: boolean
}

export function FilterDropdown({ value, options, onChange, icon = <Filter className="h-5 w-5" />, label, className, accentColor, colorAllOptions, searchable = true }: FilterDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const dropdownRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const selectedOption = options.find((o) => o.value === value) || options[0]

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 100)
    } else {
      setTimeout(() => setSearchQuery(""), 300)
    }
  }, [isOpen])

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  return (
    <div
      ref={dropdownRef}
      className={cn(
        "relative w-full min-w-[200px] rounded-2xl select-none transition-colors duration-200",
        "bg-background border border-border hover:bg-muted",
        "shadow-md shadow-black/10 dark:shadow-black/50",
        "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
        isOpen ? "rounded-3xl z-40" : "z-10",
        className
      )}
    >
      {/* Header */}
      <div
        className="flex items-center gap-4 p-3 cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-card text-primary transition-colors duration-300">
          {icon}
        </div>
        <div className="flex-1 overflow-hidden">
          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider">{label}</p>
          <h3 className="text-sm font-semibold truncate mt-0.5" style={{ color: accentColor || 'var(--color-foreground)' }}>{selectedOption.label}</h3>
        </div>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center">
          <ChevronUp
            className={cn(
              "h-5 w-5 text-muted-foreground transition-transform duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
              isOpen ? "rotate-0" : "rotate-180",
            )}
          />
        </div>
      </div>

      {/* Activity List Patterned Options */}
      <div
        className={cn(
          "absolute top-[calc(100%+8px)] left-0 right-0 z-50 rounded-3xl overflow-hidden bg-background border border-border shadow-xl shadow-black/10 dark:shadow-black/60",
          "grid",
          "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
          isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
        )}
      >
        <div className="overflow-hidden">
          <div className="px-2 pb-4 pt-2">
            
            {/* Search Input */}
            {searchable && (
              <div className="px-2 pb-2">
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    ref={searchInputRef}
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full bg-secondary/80 focus:bg-secondary text-sm rounded-xl pl-9 pr-3 py-2 outline-none border border-transparent transition-all"
                    style={{ color: colorAllOptions ? (accentColor || 'var(--color-primary)') : 'var(--color-foreground)' }}
                  />
                </div>
              </div>
            )}

            <div className="space-y-1 max-h-[260px] overflow-y-auto custom-scrollbar">
              {(() => {
                const filteredOptions = options.filter(o => 
                  o.label.toLowerCase().includes(searchQuery.toLowerCase()) || 
                  (o.description && o.description.toLowerCase().includes(searchQuery.toLowerCase()))
                );

                if (filteredOptions.length === 0) {
                  return <div className="text-center text-muted-foreground p-4 text-sm">No results found</div>;
                }

                return filteredOptions.map((option, index) => {
                const isSelected = option.value === value;
                return (
                  <div
                    key={option.value}
                    onClick={() => {
                      onChange(option.value)
                      setIsOpen(false)
                    }}
                    className={cn(
                      "flex items-center gap-3 rounded-xl p-3 cursor-pointer",
                      "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
                      "hover:bg-muted",
                      isSelected ? "bg-primary/10" : "",
                      isOpen ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
                    )}
                    style={{
                      transitionDelay: isOpen ? `${index * 50}ms` : "0ms",
                    }}
                  >
                    <div className="flex-1 min-w-0">
                      <h4 className={cn("text-sm font-semibold leading-tight")} style={{ color: (isSelected || colorAllOptions) ? (accentColor || 'var(--color-primary)') : 'var(--color-foreground)' }}>
                        {option.label}
                      </h4>
                      {option.description && (
                        <p className="text-xs text-muted-foreground truncate mt-0.5">{option.description}</p>
                      )}
                    </div>
                    {isSelected && (
                      <Check className="h-5 w-5 text-primary shrink-0 transition-opacity duration-300" />
                    )}
                  </div>
                )
              })})()}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
